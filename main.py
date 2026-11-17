import requests

dataset_slug = "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2025"
meta_url = f"https://www.data.gouv.fr/api/1/datasets/{dataset_slug}/"

dataset = requests.get(meta_url).json()

# Print all 2025 CSV files and their Resource IDs
for res in dataset["resources"]:
    if res["format"] == "csv" and "2025" in res["title"]:
        print(f"{res['title']} -> rid: {res['id']}")