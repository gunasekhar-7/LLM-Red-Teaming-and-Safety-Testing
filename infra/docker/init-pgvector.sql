-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create indexes for better performance
-- Note: IVFFlat index requires data to be present, so create after ingestion
-- CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);