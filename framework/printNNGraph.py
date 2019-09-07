import hiddenlayer as hl

graph = hl.build_graph(model, torch.zeros([1,1,64]))
graph.save('nngraph',format='png')