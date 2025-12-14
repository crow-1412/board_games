FROM node:20-alpine

WORKDIR /app/apps/web

COPY apps/web/package.json apps/web/package-lock.json* /app/apps/web/
RUN npm install

COPY apps/web /app/apps/web

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0", "--port", "3000"]
