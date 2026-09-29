FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
# apache2-utils — htpasswd для basic auth на публичном инстансе (40-basic-auth.sh)
RUN apk add --no-cache apache2-utils && : > /etc/nginx/basic-auth.conf
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.d/40-basic-auth.sh /docker-entrypoint.d/40-basic-auth.sh
RUN chmod +x /docker-entrypoint.d/40-basic-auth.sh
COPY docker-entrypoint.d/50-config.sh /docker-entrypoint.d/50-config.sh
RUN chmod +x /docker-entrypoint.d/50-config.sh
