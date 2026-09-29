"""Select LayerScale/residual dataflow, then remove only their precision round trips."""
import hashlib

def select_residuals(graph, blocks=24, width=1024, channel=757):
    import onnx
    from onnx.numpy_helper import to_array
    tensors = {t.name: t for t in graph.initializer}
    producers = {v:n for n in graph.node for v in n.output}
    consumers = {}
    for n in graph.node:
        for value in n.input:
            consumers.setdefault(value, []).append(n)
    def unwrap(value):
        while value in producers and producers[value].op_type in {'Cast','Identity'}:
            value = producers[value].input[0]
        return value
    selected, scales, rows = [], {}, []
    for block in range(blocks):
        prefix = f'/backbone/blocks.{block}/'
        norms = [n for n in graph.node if n.name.startswith(prefix) and n.op_type == 'LayerNormalization']
        candidates = []
        for n in graph.node:
            if n.name.startswith(prefix) and n.op_type == 'Mul' and len(n.input) == 2:
                gamma = [v for v in n.input if v in tensors and list(tensors[v].dims) == [width] and tensors[v].data_type == onnx.TensorProto.FLOAT]
                if len(gamma) == 1:
                    candidates.append((n,gamma[0]))
        if len(norms) != 2 or len(candidates) != 2:
            raise ValueError('Expected two actual normalizations and two vector LayerScales per block')
        state = unwrap(norms[0].input[0])
        for part, (mul,gamma) in enumerate(candidates):
            branch_value = next(v for v in mul.input if v != gamma)
            branch = producers.get(branch_value)
            if branch is None or branch.op_type != 'Add':
                raise ValueError('LayerScale must follow actual linear bias Add')
            matmuls = [producers[v] for v in branch.input if v in producers and producers[v].op_type in {'MatMul','Gemm'}]
            biases = [v for v in branch.input if v in tensors and list(tensors[v].dims) == [width]]
            if len(matmuls) != 1 or len(biases) != 1:
                raise ValueError('LayerScale branch is not verified linear+bias')
            users = consumers.get(mul.output[0], [])
            if len(users) != 1 or users[0].op_type != 'Add' or not users[0].name.startswith(prefix):
                raise ValueError('LayerScale must have one residual Add consumer')
            add = users[0]
            if len(add.input) != 2 or state not in add.input:
                raise ValueError('Residual Add does not carry previous block state')
            if part == 1 and unwrap(norms[1].input[0]) != state:
                raise ValueError('Second branch normalization not tied to first residual')
            for n in (mul, add):
                if len(n.output) != 1:
                    raise ValueError('Single-output residual operation required')
                selected.append(n.name)
                rows.append({'node':n.name,'op':n.op_type,'output':n.output[0],
                    'source_inputs':list(n.input),'source_consumers':[c.name for c in consumers.get(n.output[0],[])]})
            tensor = tensors[gamma]
            scales[gamma] = {'source_tensor_sha256':hashlib.sha256(tensor.SerializeToString()).hexdigest(), 'block':block,'part':part}
            if block == 0:
                scales[gamma].update(channel=channel,source_channel_value=float(to_array(tensor)[channel]))
            state = add.output[0]
        if block+1 < blocks:
            next_norm = next(n for n in graph.node if n.name.startswith(f'/backbone/blocks.{block+1}/') and n.op_type=='LayerNormalization')
            if unwrap(next_norm.input[0]) != state:
                raise ValueError('Residual state does not feed next block normalization')
    if len(selected) != blocks*4 or len(set(selected)) != len(selected):
        raise ValueError('Exact unique residual node coverage required')
    return selected, rows, scales

def remove_residual_roundtrips(graph, rows, blocked_ops, blocked_names):
    import onnx
    removed, proof = set(), []
    for row in rows:
        nodes = {n.name:n for n in graph.node}
        producers = {v:n for n in graph.node for v in n.output}
        original = row['output']
        selected = nodes[row['node']]
        fp32_value = selected.output[0]
        down = producers.get(original)
        def cast_to(node, dtype):
            return node is not None and node.op_type=='Cast' and len(node.input)==len(node.output)==1 and len(node.attribute)==1 and node.attribute[0].name=='to' and node.attribute[0].i==dtype
        if original in {v.name for v in graph.output}:
            up = producers.get(original)
            down = producers.get(up.input[0]) if cast_to(up, onnx.TensorProto.FLOAT) else None
            if row['source_consumers'] or not cast_to(down, onnx.TensorProto.FLOAT16) or down.input[0] != fp32_value:
                raise ValueError('Unexpected terminal external IO precision seam')
            selected.output[0] = original
            removed.update([down.name, up.name])
            proof.append({'node':row['node'],'retained_FP32_value':original,'FP32_consumers':['external_float32_output']})
            continue
        if not cast_to(down, onnx.TensorProto.FLOAT16) or down.input[0]!=fp32_value:
            raise ValueError('Expected exact selected-node FP32-to-half conversion boundary')
        linked = []
        for consumer_name in row['source_consumers']:
            consumer = nodes[consumer_name]
            if consumer.op_type in {'Shape','Size'}:
                if original not in consumer.input:
                    raise ValueError('Metadata consumer changed')
                continue
            if consumer.op_type not in blocked_ops and consumer_name not in blocked_names:
                raise ValueError('Residual state feeds unblocked arithmetic')
            found = False
            for index,value in enumerate(consumer.input):
                up = producers.get(value)
                if cast_to(up, onnx.TensorProto.FLOAT) and up.input[0]==original:
                    users = [n.name for n in graph.node if value in n.input]
                    if users != [consumer_name]:
                        raise ValueError('Cast-up has unexpected shared users')
                    consumer.input[index] = original
                    removed.add(up.name)
                    linked.append(consumer_name)
                    found = True
            if not found:
                raise ValueError('Missing exact residual half-to-FP32 consumer boundary')
        selected.output[0] = original
        removed.add(down.name)
        for value in graph.value_info:
            if value.name == original:
                value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        proof.append({'node':row['node'],'retained_FP32_value':original,'FP32_consumers':linked})
    keep = [n for n in graph.node if n.name not in removed]
    del graph.node[:]
    graph.node.extend(keep)
    live = {v for n in graph.node for v in [*n.input,*n.output]} | {v.name for v in [*graph.input,*graph.output]}
    infos = [v for v in graph.value_info if v.name in live]
    del graph.value_info[:]
    graph.value_info.extend(infos)
    types = {v.name:v.type.tensor_type.elem_type for v in [*graph.input,*graph.output,*graph.value_info]}
    types.update({t.name:t.data_type for t in graph.initializer})
    producers = {v:n for n in graph.node for v in n.output}
    for row in rows:
        node = producers[row['output']]
        expected_inputs = [onnx.TensorProto.INT64 if node.op_type == 'Expand' and index == 1 else onnx.TensorProto.FLOAT
                           for index in range(len(node.input))]
        invalid_inputs = [(index, value, types.get(value)) for index, value in enumerate(node.input)
                          if not (node.op_type == 'Clip' and index in (1, 2) and not value)
                          and types.get(value) != expected_inputs[index]]
        if node.name != row['node'] or invalid_inputs or any(types.get(v)!=onnx.TensorProto.FLOAT for v in node.output):
            raise ValueError('Selected operation has invalid FP32 arithmetic/INT64 Expand shape: ' + node.name + ' ' + repr(invalid_inputs))
        for user in graph.node:
            if row['output'] in user.input and user.op_type=='Cast' and any(a.name=='to' and a.i!=onnx.TensorProto.FLOAT for a in user.attribute):
                raise ValueError('Immediate residual downcast remains')
    return {'removed_cast_count':len(removed),'removed_cast_names':sorted(removed),'FP32_residual_flow':proof}
