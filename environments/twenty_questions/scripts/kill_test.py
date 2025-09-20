import os

from prime_cli.api.client import APIClient
from prime_cli.api.pods import PodsClient


def get_current_pod_id():
    """Find the current pod ID by matching internal IP from SSH_CONNECTION"""
    client = APIClient()
    pods_client = PodsClient(client)

    # Get current internal IP from SSH_CONNECTION
    ssh_connection = os.getenv("SSH_CONNECTION", "")
    print(f"SSH_CONNECTION: {ssh_connection}")

    if ssh_connection:
        # Format: "external_ip external_port internal_ip internal_port"
        parts = ssh_connection.split()
        if len(parts) >= 3:
            current_ip = parts[2]  # Should be 10.2.28.198
            print(f"Looking for pod with IP: {current_ip}")

            try:
                # List all pods and find matching IP
                pods = pods_client.list()
                print(f"Found {len(pods.data)} total pods")

                for pod in pods.data:
                    print(f"Pod {pod.id}: IP={pod.ip}, Status={pod.status}")

                    # Check if IP matches (handle both string and list cases)
                    if pod.ip == current_ip or (isinstance(pod.ip, list) and current_ip in pod.ip):
                        print(f"Found matching pod: {pod.id}")
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
