import onnx
from onnx import helper

def fuse_matmul_add_to_gemm(model_path, output_path):
    model = onnx.load_model(model_path)
    graph = model.graph
    nodes = graph.node

    new_nodes = []
    new_initializer = []
    i, j = 0, 0
    while i < len(nodes):
        node = nodes[i]
        if node.op_type == 'MatMul':
            assert i < len(nodes) and nodes[i + 1].op_type == 'Add'
            weight_name = str(i) + '.weight'
            bias_name = str(i) + '.bias'
            new_node = helper.make_node(
                "Gemm",
                inputs=['/' + str(j - 1) + '/Relu_output_0' if i > 0 else 'X', weight_name, bias_name],
                outputs=['/' + str(j) + '/Gemm_output_0' if i < len(nodes) - 2 else 'y_out'],
                name='/' + str(j) + '/Gemm',
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=1
            )
            W = graph.initializer[j]
            b = graph.initializer[j + 1]
            W.name = weight_name
            b.name = bias_name
            new_initializer.append(W)
            new_initializer.append(b)
            i += 2
        else:
            assert node.op_type == 'Relu'
            new_node = helper.make_node(
                "Relu",
                inputs=['/' + str(j - 1) + '/Gemm_output_0'],
                outputs=['/' + str(j) + '/Relu_output_0' if i < len(nodes) - 1 else 'y_out'],
                name='/' + str(j) + '/Relu',
            )
            i += 1
        j += 1
        
        new_nodes.append(new_node)
    
    assert len(new_initializer) == len(graph.initializer)

    X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 784])
    Y = helper.make_tensor_value_info('y_out', onnx.TensorProto.FLOAT, [1, 10])

    new_graph = helper.make_graph(new_nodes, graph.name, [X], [Y], new_initializer)
    new_model = helper.make_model(new_graph)

    onnx.checker.check_graph(new_graph)
    onnx.checker.check_model(new_model)

    print(new_nodes)

    onnx.save(new_model, output_path)

fuse_matmul_add_to_gemm("nets/mnist_relu_2_512.onnx", "nets/mnist_relu_2_512_gemm.onnx")
fuse_matmul_add_to_gemm("nets/mnist_relu_3_100.onnx", "nets/mnist_relu_3_100_gemm.onnx")
fuse_matmul_add_to_gemm("nets/mnist_relu_4_1024.onnx", "nets/mnist_relu_4_1024_gemm.onnx")
