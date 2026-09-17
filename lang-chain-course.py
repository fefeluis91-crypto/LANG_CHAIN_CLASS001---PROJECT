from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 5
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold"""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

@traceable(name="LangChain Agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"\nQuestion: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        tool_calls = ai_message.tool_calls 
        
        # Se o modelo gerou resposta em texto sem invocar ferramenta
        if not tool_calls:
            if ai_message.content:
                print(f"\nFinal Answer: {ai_message.content}")
                return ai_message.content
            else:
                # Caso o modelo feche a iteração com conteúdo vazio
                messages.append(HumanMessage(content="Synthesize the final answer based on the conversation."))
                final_resp = llm.invoke(messages)
                print(f"\nFinal Answer: {final_resp.content}")
                return final_resp.content

        # Se houver chamadas de ferramenta
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            selected_tool = tools_dict[tool_name]
            
            tool_output = selected_tool.invoke(tool_args)
            
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"]
                )
            )

if __name__ == "__main__":
    print("Iniciando o agente LangChain...")
    resultado = run_agent("What is the price of a laptop after applying a gold discount?")
    print(f"\nResultado final retornado: {resultado}")