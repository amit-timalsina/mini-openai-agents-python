## Components of an agentic system
On top of my mind i can only think of below. This will evolve as we go.

1. LLM Model
2. Agent
3. Tools
4. Orchestrator (Handoffs)
5. Memory / Session
6. Observability
7. Guardrails
8. Evaluation

## What am I doing right now?
- I am adding raw examples by using openai python sdk only.
- These examples are recreation of examples in [Openai Agent SDK](https://github.com/openai/openai-agents-python/tree/main/examples)
- I didn't want to prematurely create abstraction.
- So, first i will just write raw code then after enough examples i will start creating good abstraction and convert this into a sdk of it's own.