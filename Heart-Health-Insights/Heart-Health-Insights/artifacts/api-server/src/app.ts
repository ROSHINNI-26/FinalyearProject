import express, { type Express, type Request, type Response } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import { createProxyMiddleware } from "http-proxy-middleware";
import router from "./routes";
import { logger } from "./lib/logger";

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/api", router);

// Proxy all other requests to the Streamlit dashboard on port 8082
const STREAMLIT_PORT = process.env["STREAMLIT_PORT"] ?? "8082";
app.use(
  "/",
  createProxyMiddleware({
    target: `http://localhost:${STREAMLIT_PORT}`,
    changeOrigin: true,
    ws: true,
    logger: console,
  }),
);

export default app;
