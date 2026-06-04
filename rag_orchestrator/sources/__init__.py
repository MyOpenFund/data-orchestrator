"""Data-source connectors for RAGDataOrchestrator.

Each module here exposes a function that yields :class:`..core.SourceItem`
objects describing the documents to ingest (a local file path + a metadata
payload). Add a new file per source; the core engine stays untouched.
"""
