from google.adk.tools import ToolContext
from google.cloud import bigquery

from ..config import config

try:
    client = bigquery.Client()  # Initialize client once
except Exception as e:
    print(f"Error initializing BigQuery client: {e}")
    client = None  # Set client to None if initialization fails


def get_product_details_for_brand(tool_context: ToolContext):
    """
    Retrieves product details (title, description, attributes, and brand) from a BigQuery table for a tool_context.

    Args:
        tool_context (str): The tool_context to search for (using a LIKE '%brand%' query).

    Returns:
        str: A markdown table containing the product details, or an error message if BigQuery client initialization failed.
             The table includes columns for 'Title', 'Description', 'Attributes', and 'Brand'.
             Returns a maximum of 3 results.

    Example:
        >>> get_product_details_for_brand(tool_context)
        '| Title | Description | Attributes | Brand |\\n|---|---|---|---|\\n| Nike Air Max | Comfortable running shoes | Size: 10, Color: Blue | Nike\\n| Nike Sportswear T-Shirt | Cotton blend, short sleeve | Size: L, Color: Black | Nike\\n| Nike Pro Training Shorts | Moisture-wicking fabric | Size: M, Color: Gray | Nike\\n'
    """
    if client is None:  # Check if client initialization failed
        return "BigQuery client initialization failed. Cannot execute query."
    
    if not tool_context.user_content or not tool_context.user_content.parts:
        return "Invalid tool context or missing user content."
    
    brand = tool_context.user_content.parts[0].text

    query = f"""
        SELECT
            Title,
            Description,
            Attributes,
            Brand
        FROM
            {config.PROJECT}.{config.DATASET_ID}.{config.TABLE_ID}
        WHERE brand LIKE '%{brand}%'
        LIMIT 3
    """
    query_job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("parameter1", "STRING", brand)
        ]
    )

    query_job = client.query(query, job_config=query_job_config)
    query_job = client.query(query)
    results = query_job.result()

    markdown_table = "| Title | Description | Attributes | Brand |\n"
    markdown_table += "|---|---|---|---|\n"

    for row in results:
        title = row.Title
        description = row.Description if row.Description else "N/A"
        attributes = row.Attributes if row.Attributes else "N/A"

        markdown_table += (
            f"| {title} | {description} | {attributes} | {brand}\n"
        )

    return markdown_table