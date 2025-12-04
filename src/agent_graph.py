from agent import NormAgent, CombineAgent
import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer

class AgentGraph:
    """
    graph class of agents

    Example:
        agent_graph = AgentGraph(arg.agent_number, config.agent_type, 'deepseek')
        response = agent_graph.GetResponse(query)
    """
    def __init__(self, 
                agent_number : int,
                agent_type : list,
                llm_type : str):
        
        #! Make sure agent number is same as len of agent type
        assert agent_number == len(agent_type)
        self.agent_number = agent_number

        #* Initialize norm agent and combine agent
        self.agent_list = []
        for agent_id in range(agent_number):
            self.agent_list.append(NormAgent(agent_id, agent_type[agent_id], llm_type))
        self.combine_agent = CombineAgent(agent_number, 'Combiner', llm_type)

        self.query = ''
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def _get_sentence_embeding(self, sentence):

        """
        inner function: used to get sentence embeding

        Args:
            sentence (str): sentence string

        Returns:
            embeddings: embeding of sentence
        """

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentence)
        return embeddings
    
    def _calculate_similary(self, emb_list : list, threshold : float = 0.7) -> list:

        """
        inner function: used to calculate similarity between embedings

        Args:
            emb_list (list): sentence embeding list
            threshold (float, optional): similarity threshold. Defaults to 0.7.

        Returns:
            cluster: list of cluster
        """        

        if len(emb_list) == 1:
            return None

        emb_tensor = torch.tensor(emb_list, dtype=torch.float32)  # shape: [N, D]
        N = emb_tensor.size(0)

        # Use nn.CosineSimilarity to compute pairwise similarities
        similarity_matrix = torch.zeros(N, N)

        for i in range(N):
            e1 = emb_tensor[i].unsqueeze(0).repeat(N, 1)    # shape [N, D]
            e2 = emb_tensor                                 # shape [N, D]
            similarity_matrix[i] = self.cos(e1, e2)         # similarity with all others

        # Clustering using simple threshold logic
        visited = [False] * N
        clusters = []

        for i in range(N):
            if not visited[i]:
                cluster = [i]
                visited[i] = True
                for j in range(N):
                    if not visited[j] and similarity_matrix[i][j] >= threshold:
                        cluster.append(j)
                        visited[j] = True
                clusters.append(cluster)

        return clusters

    def GetResponse(self, query : str):

        """
        Get response based on user query

        Args:
            query (str): user's query 
        """        

        self.query = query
        agent_list = range(self.agent_number) 
        agent_response = []
        response_embeding = []

        for agent in self.agent_list :
            response = agent.get_response(query)
            response_emb = self._get_sentence_embeding(response)
            agent_response.append(response)
            response_embeding.append(response_emb)
        cluster = self._calculate_similary(response_embeding)
        agent_list = cluster

        while len(agent_list) >= 1:
            temp_agent_list = []
            response_embeding.clear()
            for agents in agent_list:
                #! combine agent
                if len(agents) > 1:
                    agent_response_combine = []
                    for agent in agents:
                        agent_response_combine.append((self.agent_list[agent], agent_response[agent]))
                    combine_response = self.combine_agent.get_response(query, agent_response_combine)
                    response_emb = self._get_sentence_embeding(combine_response)
                    response_embeding.append(response_emb)
                    temp_agent_list.append(agents)
                #! single agent
                else:
                    response = agent_response[agents[0]]
                    response_emb = self._get_sentence_embeding(response)
                    response_embeding.append(response_emb)
                    temp_agent_list.append(agents)

            cluster = self._calculate_similary(response_embeding)
            agent_list.clear()
            for group in cluster:
                cluster = []
                for agent in group:
                    cluster += agent
                agent_list.append(cluster)


            


        
        
        
        






