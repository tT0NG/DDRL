local util = {}

function util.save(filename, net, gpu)

    net:float() 
    local netsave = net:clone()
    if gpu > 0 then
        net:cuda()
    end

    for k, l in ipairs(netsave.modules) do
        if torch.type(l) == 'cudnn.SpatialConvolution' then
            local new = nn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
					      l.kW, l.kH, l.dW, l.dH, 
					      l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            netsave.modules[k] = new
        elseif torch.type(l) == 'fbnn.SpatialBatchNormalization' then
            new = nn.SpatialBatchNormalization(l.weight:size(1), l.eps, 
					       l.momentum, l.affine)
            new.running_mean:copy(l.running_mean)
            new.running_std:copy(l.running_std)
            if l.affine then
                new.weight:copy(l.weight)
                new.bias:copy(l.bias)
            end
            netsave.modules[k] = new
        end

        local m = netsave.modules[k]
        if torch.type(m.output) == 'table' then
            for i = 1,#m.output do m.output[i] = m.output[i].new() end
        else
            m.output = m.output.new()
        end
        if torch.type(m.gradInput) == 'table' then
            for i = 1,#m.gradInput do m.gradInput[i] = m.gradInput[i].new() end
        else
            m.gradInput = m.gradInput.new()
        end
        m.finput = m.finput and m.finput.new() or nil
        m.fgradInput = m.fgradInput and m.fgradInput.new() or nil
        m.buffer = nil
        m.buffer2 = nil
        m.centered = nil
        m.std = nil
        m.normalized = nil
        if m.weight then 
            m.weight = m.weight:clone()
            m.gradWeight = m.gradWeight:clone()
            m.bias = m.bias:clone()
            m.gradBias = m.gradBias:clone()
        end
    end
    if torch.type(netsave.output) == 'table' then
        for i = 1,#netsave.output do netsave.output[i] = netsave.output[i].new() end
    else
        netsave.output = netsave.output.new()
    end
    if torch.type(netsave.gradInput) == 'table' then
        for i = 1,#netsave.gradInput do netsave.gradInput[i] = netsave.gradInput[i].new() end
    else
        netsave.gradInput = netsave.gradInput.new()
    end

    netsave:apply(function(m) if m.weight then m.gradWeight = nil; m.gradBias = nil; end end)

    torch.save(filename, netsave)
end

function util.load(filename, gpu)
   local net = torch.load(filename)
   net:apply(function(m) if m.weight then 
	    m.gradWeight = m.weight:clone():zero(); 
	    m.gradBias = m.bias:clone():zero(); end end)
   return net
end

function util.cudnn(net)
    for k, l in ipairs(net.modules) do
        -- convert to cudnn
        if torch.type(l) == 'nn.SpatialConvolution' and pcall(require, 'cudnn') then
            local new = cudnn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
						 l.kW, l.kH, l.dW, l.dH, 
						 l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            net.modules[k] = new
        end
    end
    return net
end


function util.optimizeInferenceMemory(net)
    local finput, output, outputB
    net:apply(
        function(m)
            if torch.type(m):find('Convolution') then
                finput = finput or m.finput
                m.finput = finput
                output = output or m.output
                m.output = output
            elseif torch.type(m):find('ReLU') then
                m.inplace = true
            elseif torch.type(m):find('BatchNormalization') then
                outputB = outputB or m.output
                m.output = outputB
            end
    end)
end

return util
