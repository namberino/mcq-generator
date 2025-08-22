import os
import requests
# uvicorn app:app --reload --reload-exclude test/

def test_upload_files(folder_path=None):
	url = "http://127.0.0.1:8000/upload_multiple_files"
	base = r"D:/graduation_project/mcq-generator/pdfs"
	# file1 = os.path.join(base, "/Calculus2/Calculus2-1.pdf")
	# file2 = os.path.join(base, "/Calculus2/Calculus2-2.pdf")
	# file3 = os.path.join(base, "/Calculus2/Calculus2-3.pdf")
	# file4 = os.path.join(base, "/Calculus2/Calculus2-4.pdf")
	# file5 = os.path.join(base, "/Calculus2/Calculus2-5.pdf")

	calculus_folder = os.path.join(base, "Calculus2")
	files = []
	for filename in os.listdir(calculus_folder):
		if filename.endswith(".pdf"):
			file_path = os.path.join(calculus_folder, filename)
			file_name_without_suffix = os.path.splitext(filename)[0] # remove suffix
			files.append(
				("files", (file_name_without_suffix, open(file_path, "rb"))),
		)

	#? Upload multiple files manually
	# files.append(
	# 	("files", (f"{file1}", open(file2, "rb"))),
	# 	("files", (f"{file2}", open(file2, "rb"))),
	# )


	data = {"collection_name": "math", "overwrite": "True", 'qdrant_filename_prefix': "math"}
	resp = requests.post(url, files=files, data=data)
	print(resp.status_code, resp.json())

def test_generation():
    pass