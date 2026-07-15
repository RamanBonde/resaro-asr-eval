from dotenv import load_dotenv; load_dotenv()
import os, aip_sdk as aip
aip.init(os.environ['AIP_BASE_URL'], api_key=os.environ['AIP_API_KEY'])
ws = aip.Workspace.get_by_name(os.environ['AIP_WORKSPACE_NAME'])
print(f'Connected - workspace {ws.name} (id {ws.id})')