import asyncio
import websockets
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages

from dotenv import load_dotenv

load_dotenv()
model = init_chat_model("gpt-4o-mini", model_provider="openai")

trimmer = trim_messages(
    max_tokens=10,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer every question with a sarcastic joke about it in {language}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def process_message(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

graph = StateGraph(state_schema=State)
graph.add_edge(START, "model")
graph.add_node("model", process_message)

# Add memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

async def handle_client(websocket):
    try:
        async for message in websocket:
            print(f"📩 Received from client: {message}")
            config = {"configurable": {"thread_id": "abc123"}}
            language = "English"

            input_messages = [HumanMessage(message)]
            print("🔄 Generating response...")

            async for chunk, _ in app.astream(
                {"messages": input_messages, "language": language},
                config,
                stream_mode="messages"
            ):
                if isinstance(chunk, AIMessage):
                    await websocket.send(chunk.content)

    except websockets.ConnectionClosed:
        print("🔌 Client disconnected unexpectedly")

    finally:
        print("🛑 Closing WebSocket connection")

async def start_server():
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

asyncio.run(start_server())
