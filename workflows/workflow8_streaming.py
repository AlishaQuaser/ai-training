import os
import asyncio
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step, Context
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()


class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str
    response: str


class TextEvent(Event):
    delta: str


class ProgressEvent(Event):
    msg: str


class MyWorkflow(Workflow):
    def __init__(self, timeout=30, verbose=False):
        super().__init__(timeout=timeout, verbose=verbose)
        # Setup Mistral client
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set in .env file")
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-large-latest"

    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening"))
        return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        stream_response = self.client.chat.stream(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Quote the first 50 words of Meditations by Marcus Aurelius."
                }
            ]
        )

        full_response = ""
        for chunk in stream_response:
            if chunk.data.choices[0].delta.content is not None:
                delta = chunk.data.choices[0].delta.content
                ctx.write_event_to_stream(TextEvent(delta=delta))
                full_response += delta

        return SecondEvent(
            second_output="Second step complete, full response attached",
            response=full_response,
        )

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step three is happening"))
        return StopEvent(result="Workflow complete.")


async def main():
    workflow = MyWorkflow(timeout=30, verbose=False)
    handler = workflow.run(first_input="Start the workflow.")

    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.msg)
        if isinstance(ev, TextEvent):
            print(ev.delta, end="", flush=True)

    final_result = await handler
    print("\nFinal result = ", final_result)


if __name__ == "__main__":
    asyncio.run(main())
