import json
import pickle
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    topic_keywords: List[str] = field(default_factory=list)

@dataclass
class TreeNode:
    id: str
    name: str
    description: str
    level: int
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    document_chunks: Set[str] = field(default_factory=set)
    keywords: List[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None

class TextAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['would', 'could', 'should', 'might', 'may', 'must', 'can', 'will', 'shall'])
      

    
    def extract_key_phrases(self, text: str, max_phrases: int = 20) -> List[str]:
        sentences = sent_tokenize(text)
        key_phrases = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            pos_tags = pos_tag(tokens)
            
            grammar = r"""
                NP: {<DT|JJ|NN.*>+}
                    {<JJ|NN.*><IN><JJ|NN.*>}
            """
            cp = nltk.RegexpParser(grammar)
            tree = cp.parse(pos_tags)
            
            for subtree in tree:
                if isinstance(subtree, Tree) and subtree.label() == 'NP':
                    phrase = ' '.join([word for word, pos in subtree.leaves()])
                    if len(phrase.split()) >= 2 and phrase not in self.stop_words:
                        key_phrases.append(phrase)
        
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags, binary=False)
        
        for chunk in chunks:
            if isinstance(chunk, Tree):
                entity = ' '.join([token for token, pos in chunk.leaves()])
                if len(entity.split()) >= 1:
                    key_phrases.append(entity.lower())
        
        phrase_freq = Counter(key_phrases)
        return [phrase for phrase, count in phrase_freq.most_common(max_phrases)]
    
    def extract_topics_from_text(self, text: str, num_topics: int = 5) -> List[Dict[str, Any]]:
        sentences = sent_tokenize(text)
        
        if len(sentences) < num_topics:
            num_topics = max(1, len(sentences) // 2)
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        sentence_embeddings = model.encode(sentences)
        
        if len(sentences) > 1:
            kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(sentence_embeddings)
        else:
            clusters = [0]
        
        clustered_sentences = defaultdict(list)
        for i, cluster in enumerate(clusters):
            clustered_sentences[cluster].append(sentences[i])
        
        topics = []
        for cluster_id, cluster_sentences in clustered_sentences.items():
            cluster_text = ' '.join(cluster_sentences)
            key_phrases = self.extract_key_phrases(cluster_text, max_phrases=5)
            
            words = []
            for sentence in cluster_sentences:
                tokens = word_tokenize(sentence.lower())
                pos_tags = pos_tag(tokens)
                words.extend([word for word, pos in pos_tags 
                            if pos.startswith(('NN', 'JJ')) and word not in self.stop_words])
            
            word_freq = Counter(words)
            top_words = [word for word, count in word_freq.most_common(3)]
            
            topic_name = ' '.join(top_words[:2]).title() if top_words else f"Topic {cluster_id + 1}"
            
            topics.append({
                'id': f"topic_{cluster_id}",
                'name': topic_name,
                'description': cluster_sentences[0][:100] + "..." if cluster_sentences else "",
                'keywords': key_phrases,
                'sentences': cluster_sentences,
                'size': len(cluster_sentences)
            })
        
        return sorted(topics, key=lambda x: x['size'], reverse=True)

class HierarchicalTreeBuilder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.text_analyzer = TextAnalyzer()
    
    def build_tree_from_text(self, text: str, tree: 'HierarchicalTree') -> None:
        topics = self.text_analyzer.extract_topics_from_text(text, num_topics=12)
        
        if not topics:
            logger.warning("No topics extracted from text")
            return
        
        root_keywords = []
        for topic in topics[:4]:
            root_keywords.extend(topic['keywords'][:2])
        
        root_node = TreeNode(
            id="root",
            name="Document Content",
            description="Main themes and topics from the document",
            level=0,
            keywords=list(set(root_keywords))
        )
        tree.add_node(root_node)
        
    
        main_topics = topics[:6]
        
        for i, topic in enumerate(main_topics):
            node = TreeNode(
                id=topic['id'],
                name=topic['name'],
                description=topic['description'],
                level=1,
                parent_id="root",
                keywords=topic['keywords']
            )
            tree.add_node(node)
        
        if len(topics) > 6:
            remaining_topics = topics[6:]
            
            for subtopic in remaining_topics:
                subtopic_text = f"{subtopic['name']} {' '.join(subtopic['keywords'])}"
                subtopic_embedding = self.encoder.encode_text(subtopic_text)
                
                best_parent = None
                best_similarity = -1
                
                for main_topic in main_topics:
                    main_topic_text = f"{main_topic['name']} {' '.join(main_topic['keywords'])}"
                    main_topic_embedding = self.encoder.encode_text(main_topic_text)
                    
                    similarity = cosine_similarity(
                        subtopic_embedding.reshape(1, -1),
                        main_topic_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_parent = main_topic['id']
                
                if best_parent and best_similarity > 0.25:
                    subtopic_node = TreeNode(
                        id=f"{subtopic['id']}_sub",
                        name=subtopic['name'],
                        description=subtopic['description'],
                        level=2,
                        parent_id=best_parent,
                        keywords=subtopic['keywords']
                    )
                    tree.add_node(subtopic_node)
          
                    if len(subtopic['keywords']) > 3:
                        self._create_micro_topics(subtopic, subtopic_node.id, tree)
        
        logger.info(f"Built hierarchical tree with {len(tree.nodes)} nodes")
    
    def _create_micro_topics(self, parent_topic: Dict, parent_id: str, tree: 'HierarchicalTree') -> None:
        """Create micro-topics for detailed branching"""
        keywords = parent_topic['keywords']
        if len(keywords) > 4:
     
            mid_point = len(keywords) // 2
            
            micro_topic_1 = TreeNode(
                id=f"{parent_id}_micro_1",
                name=f"{keywords[0].title()} Focus",
                description=f"Detailed aspects of {keywords[0]}",
                level=3,
                parent_id=parent_id,
                keywords=keywords[:mid_point]
            )
            tree.add_node(micro_topic_1)
            
            micro_topic_2 = TreeNode(
                id=f"{parent_id}_micro_2",
                name=f"{keywords[mid_point].title()} Aspects",
                description=f"Related concepts of {keywords[mid_point]}",
                level=3,
                parent_id=parent_id,
                keywords=keywords[mid_point:]
            )
            tree.add_node(micro_topic_2)
    

class HierarchicalEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.3
        self.max_assignments = 3
    
    def encode_text(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]
    
    def classify_chunk(self, chunk: DocumentChunk, tree_nodes: Dict[str, TreeNode]) -> List[str]:
        if chunk.embeddings is None:
            chunk.embeddings = self.encode_text(chunk.content)
        
        similarities = []
        for node_id, node in tree_nodes.items():
            if node.embeddings is None:
                node_text = f"{node.description} {' '.join(node.keywords)}"
                node.embeddings = self.encode_text(node_text)
            similarity = cosine_similarity(
                chunk.embeddings.reshape(1, -1),
                node.embeddings.reshape(1, -1)
            )[0][0]
            similarities.append((node_id, similarity, node.level))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        assigned_nodes = []

        for node_id, similarity, level in similarities:
            if len(assigned_nodes) >= self.max_assignments:
                break
                
         
            level_threshold = max(0.15, self.similarity_threshold - (level * 0.05))
            
            if similarity >= level_threshold:
                should_assign = True
                for assigned_id in assigned_nodes:
                    assigned_node = tree_nodes[assigned_id]
                    current_node = tree_nodes[node_id]
                    
                
                    if (assigned_node.parent_id == node_id or 
                        current_node.parent_id == assigned_id):
                        assigned_similarity = next(s for n, s, l in similarities if n == assigned_id)
                        if abs(similarity - assigned_similarity) < 0.08:  # Reduced threshold
                            if level > assigned_node.level:
                                assigned_nodes.remove(assigned_id)
                            else:
                                should_assign = False
                                
                if should_assign:
                    assigned_nodes.append(node_id)

        return assigned_nodes if assigned_nodes else [similarities[0][0]]

class HierarchicalTree:
    def __init__(self):
        self.nodes: Dict[str, TreeNode] = {}
        self.root_nodes: Set[str] = set()

    def add_node(self, node: TreeNode) -> None:
        self.nodes[node.id] = node
        if node.parent_id is None:
            self.root_nodes.add(node.id)
        else:
            if node.parent_id in self.nodes:
                self.nodes[node.parent_id].children_ids.add(node.id)

    def get_node_path(self, node_id: str) -> List[str]:
        path = []
        current_id = node_id
        while current_id is not None:
            path.append(current_id)
            current_node = self.nodes.get(current_id)
            current_id = current_node.parent_id if current_node else None
        return list(reversed(path))

    def get_subtree_nodes(self, node_id: str) -> Set[str]:
        if node_id not in self.nodes:
            return set()
        subtree = {node_id}
        stack = [node_id]
        while stack:
            current_id = stack.pop()
            current_node = self.nodes[current_id]
            for child_id in current_node.children_ids:
                if child_id not in subtree:
                    subtree.add(child_id)
                    stack.append(child_id)
        return subtree

    def assign_chunk_to_nodes(self, chunk_id: str, node_ids: List[str]) -> None:
        for node_id in node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].document_chunks.add(chunk_id)
    def prune_tree(self, min_chunks: int = 1, prune_empty: bool = True) -> None:
        """ Remove nodes with fewer than `min_chunks` assigned chunks, and optionally remove nodes with no children and no chunks."""
        to_remove = set()

        for node_id, node in list(self.nodes.items()):
           
            if node.level == 0:
                continue

         
            if prune_empty and not node.document_chunks and not node.children_ids:
                to_remove.add(node_id)
          
            elif len(node.document_chunks) < min_chunks and not node.children_ids:
                to_remove.add(node_id)

        for node_id in to_remove:
            node = self.nodes[node_id]
            
            if node.parent_id and node.parent_id in self.nodes:
                self.nodes[node.parent_id].children_ids.discard(node_id)
            
            del self.nodes[node_id]
            

    def display_tree(self, node_id: Optional[str] = None, prefix: str = "", is_last: bool = True) -> str:
        result = []
        if node_id is None:
            root_list = list(self.root_nodes)
            for i, root_id in enumerate(root_list):
                is_last_root = (i == len(root_list) - 1)
                result.append(self.display_tree(root_id, "", is_last_root))
        else:
            node = self.nodes.get(node_id)
            if node:
                chunk_count = len(node.document_chunks)
                keywords_str = ", ".join(node.keywords[:3]) if node.keywords else "no keywords"
                
                if node.level == 0:
                    connector = "-"
                else:
                    connector = "└── " if is_last else "├── "
                
                result.append(f"{prefix}{connector}{node.name} ({chunk_count+1} chunks) [{keywords_str}]")
                
                child_prefix = prefix + ("    " if is_last else "│   ")
                children = sorted(node.children_ids)
                
                for i, child_id in enumerate(children):
                    is_last_child = (i == len(children) - 1)
                    result.append(self.display_tree(child_id, child_prefix, is_last_child))
        
        return "\n".join(result)


class AnswerGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):  # or "facebook/bart-large-cnn"
        self.generator = pipeline("text2text-generation", model=model_name)

    def generate_answer(self, query: str, context: List[str], max_length: int = 256) -> str:
        input_text = f"question: {query} context: {' '.join(context)}"
        result = self.generator(input_text, max_length=max_length, truncation=True)
        return result[0]['generated_text']
    

class QueryRouter:
    def __init__(self, encoder: HierarchicalEncoder):
        self.encoder = encoder
        self.expansion_factor = 0.7
        self.diversity_threshold = 0.8 
    
    def route_query(self, query: str, tree: HierarchicalTree, max_nodes: int = 5) -> List[str]:
        query_embedding = self.encoder.encode_text(query)
        similarities = []

        for node_id, node in tree.nodes.items():
            if node.embeddings is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    node.embeddings.reshape(1, -1)
                )[0][0]
                similarities.append((node_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = set()

        for node_id, similarity in similarities[:max_nodes * 2]:
            if len(selected_nodes) >= max_nodes:
                break
            node = tree.nodes[node_id]
            if similarity >= 0.2:
                selected_nodes.add(node_id)
                if similarity >= self.expansion_factor:
                    for child_id in node.children_ids:
                        if len(selected_nodes) < max_nodes:
                            child_node = tree.nodes[child_id]
                            if len(child_node.document_chunks) > 0:
                                selected_nodes.add(child_id)

        if not selected_nodes and similarities:
            selected_nodes.add(similarities[0][0])

        return list(selected_nodes)

    def get_filtered_chunks(self, selected_nodes: List[str], tree: HierarchicalTree, query: str) -> Set[str]:
        """NEW: Enhanced chunk filtering with query-specific ranking"""
        query_embedding = self.encoder.encode_text(query)
        chunk_candidates = set()
        
        
        for node_id in selected_nodes:
            node = tree.nodes.get(node_id)
            if node:
                chunk_candidates.update(node.document_chunks)
                if len(node.document_chunks) < 5:
                    subtree_nodes = tree.get_subtree_nodes(node_id)
                    for subtree_node_id in subtree_nodes:
                        subtree_node = tree.nodes[subtree_node_id]
                        chunk_candidates.update(subtree_node.document_chunks)
        
        return chunk_candidates

    def rank_chunks_by_query(self, chunks: Dict[str, DocumentChunk], query: str, max_chunks: int = 5) -> List[str]:
        """NEW: Rank chunks by query relevance and diversity"""
        if not chunks:
            return []
        
        query_embedding = self.encoder.encode_text(query)
        chunk_scores = []
        
        for chunk_id, chunk in chunks.items():
            if chunk.embeddings is None:
                chunk.embeddings = self.encoder.encode_text(chunk.content)
            
           
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                chunk.embeddings.reshape(1, -1)
            )[0][0]
            
            chunk_scores.append((chunk_id, similarity, chunk.embeddings))
        
   
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
       
        selected_chunks = []
        selected_embeddings = []
        
        for chunk_id, similarity, embedding in chunk_scores:
            if len(selected_chunks) >= max_chunks:
                break
            
           
            is_diverse = True
            for selected_emb in selected_embeddings:
                chunk_similarity = cosine_similarity(
                    embedding.reshape(1, -1),
                    selected_emb.reshape(1, -1)
                )[0][0]
                
                if chunk_similarity > self.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse or len(selected_chunks) == 0:
                selected_chunks.append(chunk_id)
                selected_embeddings.append(embedding)
        
        return selected_chunks

class HierarchicalRAGLayer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = HierarchicalEncoder(model_name)
        self.tree = HierarchicalTree()
        self.router = QueryRouter(self.encoder)
        self.chunks: Dict[str, DocumentChunk] = {}
        self.tree_builder = HierarchicalTreeBuilder(self.encoder)
        self.query_history = []  
        self.generator = AnswerGenerator()  

    def generate(self, query: str, max_nodes: int = 5, max_chunks: int = 5) -> Tuple[str, List[str], List[str]]:
        selected_nodes, ranked_chunks = self.search(query, max_nodes, max_chunks)
        context_chunks = [self.chunks[cid].content for cid in ranked_chunks]
        generated_answer = self.generator.generate_answer(query, context_chunks)
        return generated_answer, selected_nodes, ranked_chunks
    

    def build_tree_from_document(self, document_text: str) -> None:
        self.tree_builder.build_tree_from_text(document_text, self.tree)
        
        sentences = sent_tokenize(document_text)
        
     
        total_sentences = len(sentences)
        if total_sentences > 50:
            chunk_size = 3  # Smaller chunks 
        elif total_sentences > 20:
            chunk_size = 4  # Medium chunks
        else:
            chunk_size = 5  # Larger chunks 
        
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_content = ' '.join(chunk_sentences)
            
         
            chunk_keywords = self.tree_builder.text_analyzer.extract_key_phrases(chunk_content, max_phrases=5)
            
            chunk = DocumentChunk(
                id=f"chunk_{i // chunk_size + 1}",
                content=chunk_content,
                metadata={
                    "sentence_range": f"{i+1}-{min(i+chunk_size, len(sentences))}",
                    "chunk_size": len(chunk_sentences),
                    "word_count": len(chunk_content.split())
                },
                topic_keywords=chunk_keywords
            )
            self.add_document_chunk(chunk)

    def add_document_chunk(self, chunk: DocumentChunk) -> None:
        self.chunks[chunk.id] = chunk
        assigned_nodes = self.encoder.classify_chunk(chunk, self.tree.nodes)
        self.tree.assign_chunk_to_nodes(chunk.id, assigned_nodes)
        logger.info(f"Chunk '{chunk.id}' assigned to nodes: {assigned_nodes}")

    def search(self, query: str, max_nodes: int = 5, max_chunks: int = 5) -> Tuple[List[str], List[str]]:
        """NEW: Enhanced search with query-specific ranking"""
  
        self.query_history.append(query)
        
        
        selected_nodes = self.router.route_query(query, self.tree, max_nodes)
        
       
        candidate_chunks = self.router.get_filtered_chunks(selected_nodes, self.tree, query)
        
  
        filtered_chunks_dict = {cid: self.chunks[cid] for cid in candidate_chunks if cid in self.chunks}
        
       
        ranked_chunks = self.router.rank_chunks_by_query(filtered_chunks_dict, query, max_chunks)
        
        logger.info(f"Query '{query}' routed to nodes: {selected_nodes}")
        logger.info(f"Ranked to {len(ranked_chunks)} chunks: {ranked_chunks}")
        
        return selected_nodes, ranked_chunks

    def get_chunk_content(self, chunk_ids: List[str]) -> List[DocumentChunk]:
        """Return chunks in the order they were ranked"""
        return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]

    def save_system(self, filepath: str) -> None:
        data = {
            'tree_nodes': self.tree.nodes,
            'root_nodes': self.tree.root_nodes,
            'chunks': self.chunks,
            'query_history': self.query_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_system(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.tree.nodes = data['tree_nodes']
        self.tree.root_nodes = data['root_nodes']
        self.chunks = data['chunks']
        self.query_history = data.get('query_history', [])

def get_user_input():
    print("=" * 60)
    print("HIERARCHICAL RAG SYSTEM")
    print("=" * 60)
    print("\nPlease paste your document/paragraph below.")
    print("You can paste multiple paragraphs - just keep typing.")
    print("When finished, type 'END' on a new line and press Enter.\n")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return None
    
    document_text = '\n'.join(lines).strip()
    
    if not document_text:
        print("No text provided. Exiting...")
        return None
    
    return document_text

def interactive_query_session(rag_system):
    print("\n" + "=" * 60)
    print("QUERY SESSION")
    print("=" * 60)
    print("You can now ask questions about your document.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                print("Please enter a question.\n")
                continue
            
            print(f"\nProcessing query: '{query}'")
            print("-" * 40)

            generated_answer, selected_nodes, ranked_chunks = rag_system.generate(query)

            print(f"\nSelected nodes: {[rag_system.tree.nodes[nid].name for nid in selected_nodes]}")
            print(f"\nGenerated Answer:\n{generated_answer}\n")

            print("Context chunks used:")
            for i, chunk_id in enumerate(ranked_chunks, 1):
                chunk = rag_system.chunks[chunk_id]
                print(f"\n{i}. [Chunk {chunk.id}] {chunk.content}")
                if chunk.topic_keywords:
                    print(f"   Keywords: {', '.join(chunk.topic_keywords[:3])}")

            print("\n" + "=" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

def main():
    document_text = get_user_input()
    
    if document_text is None:
        return
    
    print(f"\nReceived document with {len(document_text)} characters.")
    print("Processing document...")
    
    rag_system = HierarchicalRAGLayer()
    
    print("Building hierarchical tree from document...")
    try:
        rag_system.build_tree_from_document(document_text)
        rag_system.tree.prune_tree(min_chunks=2, prune_empty=True)

    except Exception as e:
        print(f"Error processing document: {e}")
        print("Please make sure your text contains meaningful content.")
        return
    
    print("\n" + "=" * 50)
    print("HIERARCHICAL TREE STRUCTURE")
    print("=" * 50)
    print(rag_system.tree.display_tree())
    
    save_option = input("\nWould you like to save this system for later use? (y/n): ").strip().lower()
    if save_option == 'y':
        filename = input("Enter filename (without extension): ").strip()
        if filename:
            try:
                rag_system.save_system(f"{filename}.ragl")
                print(f"System saved as {filename}.ragl")
            except Exception as e:
                print(f"Error saving system: {e}")
    
    interactive_query_session(rag_system)

if __name__ == "__main__":
    main()