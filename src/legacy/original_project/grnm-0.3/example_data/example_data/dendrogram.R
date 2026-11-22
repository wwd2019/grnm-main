library(dplyr)
library(ggraph)
library(igraph)
library(stringi)

root_list <- read.table('/sibcb2/zenganlab2/linyaru/pando.GRN/tree_building_code/TF_for_tree.txt', head=FALSE)

root_list <- root_list$V1
root_list <- c('NFYC')
# root <- 'RFX3'
for(root in root_list){
#   vertices_path = paste('/sibcb2/zenganlab2/linyaru/pando.GRN/tree_building_code/res/', root, '_nodes.csv', sep = '')
#   neg_edges_path = paste('/sibcb2/zenganlab2/linyaru/pando.GRN/tree_building_code/res/', root, '_neg_edge.csv', sep = '')
#   pos_edges_path = paste('/sibcb2/zenganlab2/linyaru/pando.GRN/tree_building_code/res/', root, '_pos_edge.csv', sep = '')
  vertices_path = paste('./', root, '_nodes.csv', sep = '')
  neg_edges_path = paste('./', root, '_neg_edge.csv', sep = '')
  pos_edges_path = paste('./', root, '_pos_edge.csv', sep = '')
  vertices <- read.table(vertices_path, header=T, sep=',') #导入节点数据
  neg_edges <- read.table(neg_edges_path, header=T, sep=',',row.names = 1)
  pos_edges <- read.table(pos_edges_path, header=T, sep=',',row.names = 1)
  neg_edges["attr"] = "negtaive"
  pos_edges["attr"] = "possitive"
  edges = full_join(neg_edges, pos_edges)
  name <- unique(c(as.character(edges$from), as.character(edges$to)))
  showname <- sapply(name, function(i){
    if(startsWith(i,"SMESG")){
      return("")
    }else{
      return(i)
    }
  })
  tf <- sapply(name, function(i){
    if(startsWith(i,"SMESG")){
      return(FALSE)
    }else{
      return(TRUE)
    }
  })
  vertices <- data.frame(
    name=name,
    showname = showname
  )
  mygraph <- graph_from_data_frame(edges,vertices = vertices)
  # ggraph(mygraph, layout = 'dendrogram') + geom_edge_diagonal()
#   p <- ggraph(mygraph, layout = 'tree', circular = TRUE) +
#     geom_edge_diagonal(aes(colour = factor(attr))) +
#     scale_edge_color_manual(values = c("cornflowerblue","salmon")) +
#     geom_node_text(aes(label= showname)) +
#     geom_node_point(shape = 21,fill = "grey",color = "black", size = 1)
#
# 假设 mygraph 已经是 igraph 或 tidygraph 对象
layout <- create_layout(mygraph, layout = "tree", circular = TRUE)

# 计算每个节点的角度
layout$angle <- atan2(layout$y, layout$x) * 180 / pi
layout$hjust <- ifelse(layout$angle < -90 | layout$angle > 90, 1, 0)
layout$angle <- ifelse(layout$angle < -90 | layout$angle > 90, layout$angle + 180, layout$angle)

p<-ggraph(layout) +
  geom_edge_diagonal(aes(colour = factor(attr))) +
  scale_edge_color_manual(values = c("cornflowerblue", "salmon")) +
  geom_node_point(shape = 21, fill = "grey", color = "black", size = 1) +
  geom_node_text(aes(label = showname, angle = angle, hjust = hjust), size = 3)

#   file_name = paste('/sibcb2/zenganlab2/linyaru/pando.GRN/tree_building_code/res/pic/',root,'_tree_pic.pdf', sep='')
    file_name = paste('./pic/',root,'_tree_pic.pdf', sep='')
  ggsave(p,filename = file_name,width = 9,height = 7)
}


