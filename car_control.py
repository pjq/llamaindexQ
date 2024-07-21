import logging
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from llm_utils import load_and_initialize_llm
from llama_index.core.agent import ReActAgent
import random


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to enable mock functions
USE_MOCK = True

# Real function to turn the car
def turn_car(direction: str, duration: int) -> str:
    """Turn the car in the specified direction for a given duration."""
    if direction not in ["left", "right", "up", "down"]:
        logger.error("Invalid direction: %s", direction)
        return "Invalid direction. Please specify 'left', 'right', 'up', or 'down'."
    logger.info("Turning car %s for %d seconds", direction, duration)
    return f"Car turned {direction} for {duration} seconds."

# Mock function to turn the car
def mock_turn_car(direction: str, duration: int) -> str:
    """Mock turn the car in the specified direction for a given duration."""
    logger.info("Mock: Turning car %s for %d seconds", direction, duration)
    return f"Mock: Car would turn {direction} for {duration} seconds."

# Real function to check the car camera
def check_camera() -> str:
    """Check what the car camera sees."""
    # Placeholder for actual camera check logic
    logger.info("Checking camera view")
    return "Camera view: [image data]"

# Mock function to check the car camera
def mock_check_camera() -> str:
    """Mock check what the car camera sees."""
    logger.info("Mock: Checking camera view")

    scenes = [
        "Camera view: [image data of a sunny day]",
        "Camera view: [image data of a rainy day]",
        "Camera view: [image data of a night scene]",
        "Camera view: [image data of a busy street]",
        "Camera view: [image data of a quiet park]"
    ]

    return random.choice(scenes)

# Select functions based on the flag
turn_car_fn = mock_turn_car if USE_MOCK else turn_car
check_camera_fn = mock_check_camera if USE_MOCK else check_camera

# Define the tools using FunctionTool
turn_car_tool = FunctionTool.from_defaults(
    turn_car_fn,
    name="TurnCarTool",
    description="A tool to turn the car in specified direction for a given duration."
)

check_camera_tool = FunctionTool.from_defaults(
    check_camera_fn,
    name="CheckCameraTool",
    description="A tool to check what the car camera sees."
)

# Combine tools into a single agent
llm = load_and_initialize_llm()
tools = [turn_car_tool, check_camera_tool]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# ChatBot interface for car control
def chatbot_interface():
    print("Welcome to the Car Control ChatBot!")
    while True:
        user_input = input("Enter command (turn/check/exit): ").strip().lower()
        if user_input == "exit":
            print("Exiting Car Control ChatBot. Goodbye!")
            break
        else:
            response = agent.chat(user_input)
            print(response)

if __name__ == "__main__":
    chatbot_interface()