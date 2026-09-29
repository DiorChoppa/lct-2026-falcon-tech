import cameras from "../../data/route-cameras.json";
import type { Camera, Registry } from "./types";

// JSON строкой: kind/zone в нём — просто string, форму гарантирует генератор
// (web/scripts/route_cameras.py) и тест registry.test.ts.
export const REGISTRY = cameras as unknown as Registry;
export const CAMERAS_BY_ID = new Map<string, Camera>(REGISTRY.cameras.map((c) => [c.id, c]));
