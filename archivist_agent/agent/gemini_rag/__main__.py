"""Main entry point for the case agent."""

import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv

from aigency.agents.generator import AgentA2AGenerator
from aigency.utils.config_service import ConfigService
from aigency.utils.logger import Logger, get_logger

load_dotenv()


def main():

    config = {
        "log_level": "DEBUG",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_file": "app.log",
        "logger_name": "aigency",
    }
    logger = Logger(config=config)
    logger.info("Logger initialized")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "agent_config.yaml")

        config_service = ConfigService(config_file=config_path)
        agent_config = config_service.config

        agent = AgentA2AGenerator.create_agent(agent_config=agent_config)
        agent_card = AgentA2AGenerator.build_agent_card(agent_config=agent_config)
        executor = AgentA2AGenerator.build_executor(agent=agent, agent_card=agent_card)
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
        logger.info(f"Server object created: {server}")
        logger.info("🚀 Starting case agent ...")
        uvicorn.run(server.build(), host="0.0.0.0", port=8080)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger = get_logger()
        logger.info("Application interrupted by user. Exiting...")
