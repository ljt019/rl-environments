import os

from prime_cli.api.client import APIClient
from prime_cli.api.pods import PodsClient


def get_current_pod_id():
    """Find the current pod ID by matching various criteria"""
    client = APIClient()
    pods_client = PodsClient(client)

    # Get environment info
    ssh_connection = os.getenv("SSH_CONNECTION", "")
    jupyter_password = os.getenv("JUPYTER_PASSWORD", "")
    print(f"SSH_CONNECTION: {ssh_connection}")
    print(f"JUPYTER_PASSWORD: {jupyter_password}")

    # Get hostname
    try:
        import subprocess

        hostname = subprocess.check_output(["hostname"], text=True).strip()
        print(f"HOSTNAME: {hostname}")
    except:
        hostname = ""

    try:
        # List all pods
        pods = pods_client.list()
        print(f"Found {len(pods.data)} total pods")

        for pod in pods.data:
            print(f"\nPod {pod.id}:")
            print(f"  IP: {pod.ip}")
            print(f"  Status: {pod.status}")
            print(f"  Name: {pod.name}")
            if hasattr(pod, "jupyter_password"):
                print(f"  Jupyter Password: {pod.jupyter_password}")

            # Try multiple matching strategies

            # Strategy 1: Match by Jupyter password
            if jupyter_password and hasattr(pod, "jupyter_password") and pod.jupyter_password == jupyter_password:
                print(f"Found matching pod by Jupyter password: {pod.id}")
                return pod.id

            # Strategy 2: Match by hostname in pod name/ID
            if hostname and (hostname in pod.id or (pod.name and hostname in pod.name)):
                print(f"Found matching pod by hostname: {pod.id}")
                return pod.id

            # Strategy 3: If only one ACTIVE pod, assume it's this one
            if pod.status == "ACTIVE":
                print(f"Found ACTIVE pod (assuming current): {pod.id}")
                return pod.id

    except Exception as e:
        print(f"Error listing pods: {e}")
        return None

    print("Could not determine current pod ID")
    return None


def kill_current_pod():
    """Kill the current pod"""
    pod_id = get_current_pod_id()

    if not pod_id:
        print("Cannot kill pod - unable to identify current pod ID")
        return False

    try:
        client = APIClient()
        pods_client = PodsClient(client)
        pods_client.delete(pod_id=pod_id)
        print(f"Successfully killed pod '{pod_id}'")
        return True
    except Exception as e:
        print(f"Failed to kill pod '{pod_id}': {e}")
        return False


if __name__ == "__main__":
    kill_current_pod()
