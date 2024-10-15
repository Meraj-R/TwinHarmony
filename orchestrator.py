# orchestrator.py
import asyncio
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

import aiosqlite
import boto3
import numpy as np
import paho.mqtt.client as mqtt
import redis
import tensorflow as tf
import uvicorn
import uvloop
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from kazoo.client import KazooClient
from opencensus.ext.fastapi import FastApiInstrumentation
from opencensus.trace.samplers import ProbabilitySampler
from tensorflow.keras import layers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
DB_FILE = "factory_twin.db"
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
AWS_REGION = "us-west-2"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
ZOOKEEPER_HOSTS = "localhost:2181"


@dataclass
class MachineConfig:
    id: str
    max_temp: float
    max_vibration: float
    max_pressure: float
    optimal_rpm: float
    production_rate: float
    location: tuple
    node_id: str  # Distributed node assignment


class MachineLearningModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential(
            [
                layers.Dense(128, activation="relu", input_shape=(5,)),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    async def predict(self, data):
        return float(self.model.predict(data.reshape(1, -1), verbose=0)[0][0])


class MachineDigitalTwin:
    def __init__(self, config, ml_model, redis_client):
        self.config = config
        self.ml_model = ml_model
        self.active = False
        self.data_buffer = []
        self.redis = redis_client
        self.mqtt_client = mqtt.Client(client_id=f"machine_{config.id}")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
        self.s3 = boto3.client("s3", region_name=AWS_REGION)
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        client.subscribe(f"factory/{self.config.id}/control")
        logging.info(f"Connected MQTT for {self.config.id}")

    async def simulate(self):
        while self.active:
            data = self._generate_sensor_data()
            await self._process_data(data)
            await asyncio.sleep(0.1)

    def _generate_sensor_data(self):
        return {
            "timestamp": datetime.now().isoformat(),
            "temperature": np.random.normal(self.config.max_temp * 0.7, 5),
            "vibration": np.random.normal(self.config.max_vibration * 0.6, 0.5),
            "pressure": np.random.normal(self.config.max_pressure * 0.8, 5),
            "rpm": np.random.normal(self.config.optimal_rpm, 50),
            "production": np.random.normal(self.config.production_rate, 2),
        }

    async def _process_data(self, data):
        features = np.array(
            [
                data["temperature"],
                data["vibration"],
                data["pressure"],
                data["rpm"],
                data["production"],
            ]
        )
        failure_prob = await self.ml_model.predict(features)
        data["health_score"] = max(0, 100 - (failure_prob * 100))
        self.data_buffer.append(data)

        self.mqtt_client.publish(f"factory/{self.config.id}/data", json.dumps(data))
        self.redis.rpush(f"data:{self.config.id}", json.dumps(data))
        self.redis.ltrim(f"data:{self.config.id}", -100, -1)

        asyncio.get_event_loop().run_in_executor(
            self.executor, self._upload_to_s3, data
        )
        await self._save_to_db(data)

    def _upload_to_s3(self, data):
        bucket = "factory-twin-data"
        key = f"{self.config.id}/{data['timestamp']}.json"
        self.s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data))

    async def _save_to_db(self, data):
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute(
                """INSERT INTO sensor_data 
                (machine_id, timestamp, temperature, vibration, pressure, rpm, 
                production, health_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.config.id,
                    data["timestamp"],
                    data["temperature"],
                    data["vibration"],
                    data["pressure"],
                    data["rpm"],
                    data["production"],
                    data["health_score"],
                ),
            )
            await db.commit()


class DigitalTwinOrchestrator:
    def __init__(self):
        self.app = FastAPI()
        self.machines = {}
        self.ml_model = MachineLearningModel()
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        self.zk = KazooClient(hosts=ZOOKEEPER_HOSTS)
        self.node_id = str(uuid.uuid4())
        self.zk.start()
        self.setup_database()
        self.setup_machines()
        self.setup_routes()
        FastApiInstrumentation().instrument_app(
            self.app, sampler=ProbabilitySampler(1.0)
        )

    def setup_database(self):
        asyncio.run(self._create_tables())

    async def _create_tables(self):
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute(
                """CREATE TABLE IF NOT EXISTS sensor_data
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 machine_id TEXT,
                 timestamp TEXT,
                 temperature REAL,
                 vibration REAL,
                 pressure REAL,
                 rpm REAL,
                 production REAL,
                 health_score REAL)"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS nodes
                (node_id TEXT PRIMARY KEY,
                 status TEXT,
                 last_heartbeat TIMESTAMP)"""
            )
            await db.commit()

    def setup_machines(self):
        configs = [
            MachineConfig(
                "CNC_1", 80.0, 5.0, 100.0, 1500.0, 10.0, (0, 0, 0), self.node_id
            ),
            MachineConfig(
                "Press_1", 70.0, 4.0, 120.0, 1200.0, 15.0, (10, 0, 0), self.node_id
            ),
            MachineConfig(
                "Welder_1", 90.0, 3.0, 80.0, 1000.0, 8.0, (20, 0, 0), self.node_id
            ),
        ]
        for config in configs:
            machine = MachineDigitalTwin(config, self.ml_model, self.redis)
            self.machines[config.id] = machine
            self.zk.ensure_path(f"/twins/{config.id}")
            self.zk.set(
                f"/twins/{config.id}", json.dumps({"node": self.node_id}).encode()
            )

    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            return self._get_html_template()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket):
            await websocket.accept()
            while True:
                await self._send_websocket_data(websocket)
                await asyncio.sleep(1)

    def _get_html_template(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Orchestrated Digital Twin</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.1/socket.io.js"></script>
        </head>
        <body>
            <h1>Orchestrated Factory Digital Twin</h1>
            <div id="charts" style="display: flex; flex-wrap: wrap;"></div>
            <div id="orchestration">Nodes: <span id="node-count">0</span></div>
            <script>
                const socket = io('http://localhost:8000');
                socket.on('machine_data', (data) => {
                    Object.keys(data).forEach(machine => {
                        let div = document.getElementById(machine);
                        if (!div) {
                            div = document.createElement('div');
                            div.id = machine;
                            div.style.width = '30%';
                            div.style.margin = '1%';
                            document.getElementById('charts').appendChild(div);
                        }
                        Plotly.newPlot(machine, [{
                            x: data[machine].map(d => d.timestamp),
                            y: data[machine].map(d => d.health_score),
                            name: 'Health',
                            type: 'scatter'
                        }], {title: machine + ' Health Score'});
                    });
                });
                socket.on('orchestration', (data) => {
                    document.getElementById('node-count').textContent = data.nodes;
                });
            </script>
        </body>
        </html>
        """

    async def _send_websocket_data(self, websocket):
        data = {
            mid: [json.loads(d) for d in self.redis.lrange(f"data:{mid}", -10, -1)]
            for mid in self.machines.keys()
        }
        await websocket.send_json({"type": "machine_data", "data": data})
        nodes = len(self.zk.get_children("/nodes"))
        await websocket.send_json({"type": "orchestration", "nodes": nodes})

    async def heartbeat(self):
        while True:
            await self._update_heartbeat()
            self.zk.ensure_path(f"/nodes/{self.node_id}")
            self.zk.set(f"/nodes/{self.node_id}", b"active")
            await asyncio.sleep(5)

    async def _update_heartbeat(self):
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute(
                """INSERT OR REPLACE INTO nodes 
                (node_id, status, last_heartbeat) VALUES (?, ?, ?)""",
                (self.node_id, "active", datetime.now().isoformat()),
            )
            await db.commit()

    async def start(self):
        for machine in self.machines.values():
            machine.active = True
            asyncio.create_task(machine.simulate())

        for machine in self.machines.values():
            machine.mqtt_client.loop_start()

        asyncio.create_task(self.heartbeat())


if __name__ == "__main__":

    orchestrator = DigitalTwinOrchestrator()

    async def main():
        await orchestrator.start()
        config = uvicorn.Config(orchestrator.app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        await server.serve()

    uvloop.install()
    asyncio.run(main())
