import sys
sys.path.insert(0, 'src')
from rag import RAGEngine

r = RAGEngine()
res = r.query('probleme de connexion portail web')
print('--- REPONSE LLM ---')
print(res['answer'])
print()
print('--- SOURCES ---')
for s in res['sources'][:3]:
    print(f"  ticket={s['ticket_id']} sim={s['similarity']:.1%} product={s['product']}")
