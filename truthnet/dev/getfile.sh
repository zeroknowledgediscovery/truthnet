gsutil -m cp -r gs://veritas-experiment ./
gsutil ls -l gs://veritas-experiment | grep -v TOTAL | grep -v README > filetimes.csv
