import random
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.utils.workflow import draw_all_possible_flows
import os


class BranchA1Event(Event):
    payload: str


class BranchA2Event(Event):
    payload: str


class BranchB1Event(Event):
    payload: str


class BranchB2Event(Event):
    payload: str


class BranchWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> BranchA1Event | BranchB1Event:
        if random.randint(0, 1) == 0:
            print("Go to branch A")
            return BranchA1Event(payload="Branch A")
        else:
            print("Go to branch B")
            return BranchB1Event(payload="Branch B")

    @step
    async def step_a1(self, ev: BranchA1Event) -> BranchA2Event:
        print(ev.payload)
        return BranchA2Event(payload=ev.payload)

    @step
    async def step_b1(self, ev: BranchB1Event) -> BranchB2Event:
        print(ev.payload)
        return BranchB2Event(payload=ev.payload)

    @step
    async def step_a2(self, ev: BranchA2Event) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Branch A complete.")

    @step
    async def step_b2(self, ev: BranchB2Event) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Branch B complete.")


async def main():
    branch_workflow = BranchWorkflow(timeout=10, verbose=False)

    result = await branch_workflow.run(first_input="Start the workflow.")
    print(result)

    os.makedirs("workflows", exist_ok=True)

    WORKFLOW_FILE = "workflows/branching.html"
    draw_all_possible_flows(branch_workflow, filename=WORKFLOW_FILE)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
