from agents import Agent, Runner
import agentops

agentops.init()  # Loads key from your .env

agent = Agent(
    name="Chat Insight Agent",
    instructions="You are a helpful assistant for interpreting user analytics queries."
)

result = Runner.run_sync(agent, "Compare walk-in leads with online leads.")
print("LLM Output:", result.final_output) 