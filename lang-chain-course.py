from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 5
MODEL = "qwen2.5:1.5b"


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}

    # Trata variações em inglês/singular/plural
    prod = product.lower().strip()
    if prod == "headphone":
        prod = "headphones"

    return prices.get(prod, 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price and return the final price.

    Available tiers: bronze, silver, gold
    """
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier.lower(), 0)
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
                "You are an intelligent shopping assistant.\n"
                "You have access to get_product_price and apply_discount tools.\n\n"
                "SMART CATALOG MAPPING & FUZZY MATCHING:\n"
                "1. The catalog tools strictly accept only these exact English product names:\n"
                "   - 'laptop'\n"
                "   - 'headphones'\n"
                "   - 'keyboard'\n"
                "2. BEFORE calling get_product_price, you MUST map any user input (including typos, "
                "synonyms, or Portuguese words like 'computador', 'computadore', 'lapto', 'fone', 'teclado') "
                "to the closest valid catalog item.\n"
                "   - Example: 'computador', 'coitadore', 'notebook', 'lapto' -> map to 'laptop'\n"
                "   - Example: 'fone', 'headphone', 'foninho' -> map to 'headphones'\n"
                "   - Example: 'teclado', 'keybord' -> map to 'keyboard'\n\n"
                "3. Always invoke get_product_price with the corrected English catalog name.\n"
                "4. Call apply_discount AFTER getting the valid price.\n"
                "5. Do not invent prices or do math yourself."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        tool_calls = ai_message.tool_calls

        # 1. CASO O MODELO NÃO CHAME FERRAMENTA NO TURNO
        if not tool_calls:
            # Trava para modelos pequenos: se pediu desconto mas ele não chamou a tool de desconto ainda, força a chamada
            has_discount_call = any(
                isinstance(m, ToolMessage) and m.name == "apply_discount"
                for m in messages
            )
            if "discount" in question.lower() and not has_discount_call:
                messages.append(
                    HumanMessage(
                        content="Now apply the requested discount tier using the apply_discount tool on the price you just found."
                    )
                )
                continue

            # Se gerou resposta com texto final
            if ai_message.content:
                print(f"\nFinal Answer: {ai_message.content}")
                return ai_message.content
            else:
                # Caso feche a iteração vazia
                messages.append(
                    HumanMessage(
                        content="Synthesize the final answer based on the conversation."
                    )
                )
                final_resp = llm.invoke(messages)
                print(f"\nFinal Answer: {final_resp.content}")
                return final_resp.content

        # 2. CASO O MODELO CHAME FERRAMENTA(S)
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Tratamento para Pydantic / Dicionários aninhados em LLMs menores
            for key, value in tool_args.items():
                if isinstance(value, dict) and "type" in value:
                    tool_args[key] = value["type"]
                elif isinstance(value, dict) and "value" in value:
                    tool_args[key] = value["value"]

            selected_tool = tools_dict[tool_name]
            tool_output = selected_tool.invoke(tool_args)

            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )

            # Trava para o Ollama: executa apenas 1 ferramenta por turno para garantir a ordem sequencial
            break


if __name__ == "__main__":
    print("Iniciando o agente LangChain...")
    resultado = run_agent(
        "What is the price of a headphones after applying a silver discount?"
    )
    print(f"\nResultado final retornado: {resultado}")