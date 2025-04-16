import asyncio
import random
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step, Context


class StepTwoEvent(Event):
    query: str


class StepThreeEvent(Event):
    result: str


class ConcurrentFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> StepTwoEvent:
        # Send multiple events in parallel to StepTwo
        ctx.send_event(StepTwoEvent(query="Query 1"))
        ctx.send_event(StepTwoEvent(query="Query 2"))
        ctx.send_event(StepTwoEvent(query="Query 3"))

        return StepTwoEvent(query="Dummy query to satisfy return")

    @step(num_workers=4)
    async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StepThreeEvent:
        print(f"Running query: {ev.query}")
        await asyncio.sleep(random.randint(1, 5))

        return StepThreeEvent(result=ev.query)

    @step
    async def step_three(self, ctx: Context, ev: StepThreeEvent) -> StopEvent:
        result = ctx.collect_events(ev, [StepThreeEvent] * 3)
        if result is None:
            print("Not all events received yet.")
            return None

        print(result)
        return StopEvent(result="Done")


async def main():
    w = ConcurrentFlow(timeout=10, verbose=False)

    result = await w.run(message="Start the workflow.")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
