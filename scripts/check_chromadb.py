import sys, traceback
print('Python executable:', sys.executable)

try:
    import importlib
    spec = importlib.util.find_spec('chromadb')
    print('chromadb spec:', spec)
except Exception as e:
    print('Error finding spec for chromadb:', e)

try:
    import chromadb
    print('Imported chromadb OK; version:', getattr(chromadb, '__version__', 'unknown'))
    print('Has PersistentClient:', hasattr(chromadb, 'PersistentClient'))
except Exception as e:
    print('Failed to import chromadb:')
    traceback.print_exc()

try:
    import chromadb.config as cfg
    print('Imported chromadb.config OK; has Settings:', hasattr(cfg, 'Settings'))
except Exception as e:
    print('Failed to import chromadb.config:')
    traceback.print_exc()

try:
    # test PersistentClient construction minimal (do not create files)
    if hasattr(chromadb, 'PersistentClient'):
        try:
            client = chromadb.PersistentClient(path='./.chromadb_test')
            print('PersistentClient constructed OK')
        except Exception as e:
            print('PersistentClient construction failed:')
            traceback.print_exc()
except Exception:
    pass

print('Done')
