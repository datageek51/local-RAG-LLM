import sys
import traceback
import pathlib

# Ensure repo root is on sys.path so we can import Utilities.py from project root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

print('Python executable:', sys.executable)

try:
    from Utilities import initialize_chroma_db
    docs = [
        "This is a short test document about cats and dogs.",
        "This is another document discussing gardening and plants."
    ]
    print('Calling initialize_chroma_db with', len(docs), 'documents...')
    coll = initialize_chroma_db(docs)
    print('initialize_chroma_db returned collection object of type:', type(coll))
    try:
        print('Trying a sample query ("cats")...')
        res = coll.query(query_texts=['cats'], n_results=2)
        print('Query result (keys):', list(res.keys()))
        # print a small excerpt of results for inspection
        print('Documents returned:', res.get('documents'))
        print('IDs returned:', res.get('ids'))
    except Exception as qe:
        print('Query failed with error:', qe)
        traceback.print_exc()

except Exception as e:
    print('Import/init failed:', e)
    traceback.print_exc()
