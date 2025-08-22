from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

db = {
    "a": 100,
    "b": 200,
}


@mcp.tool
def get_value_from_a(key: str) -> str:
    """
    Get the value of a from the database
    """
    return {"value": db[key]}

@mcp.tool
def get_value_from_b(key: str) -> str:
    """
    Get the value of b from the database
    """
    return {"value": db[key]}

@mcp.tool
def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers
    """
    return a + b



if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)
