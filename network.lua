require 'dpnn'
require 'rnn'
require 'cutorch'
require 'nngraph'
require 'optim'
require 'image'

require 'VRMSEReward'
require 'SpatialGlimpse_inverse'
util = paths.dofile('util.lua')
-- nngraph.setDebug(true)
opt = lapp[[
   -b,--batchSize             (default 32)        
   -r,--lr                    (default 0.0002)    

   --dataset                  (default 'folder')  
   --nThreads                 (default 4)         

   --beta1                    (default 0.5)       
   --ntrain                   (default math.huge) 
   --display                  (default 0)         
   --display_id               (default 10)        
   --gpu                      (default 1)         
   --GAN_loss_after_epoch     (default 5)
   --name                     (default 'fullmodel')
   --checkpoints_name         (default '')        
   --checkpoints_epoch        (default 0)         
   --epoch                    (default 1)         
   --nc                       (default 3)         

   --niter                    (default 250)  

   --rewardScale              (default 1)     
   --rewardAreaScale          (default 4)     
   --locatorStd               (default 0.11)  

   --glimpseHiddenSize        (default 128)  
   --glimpsePatchSize         (default '60,45')
   --glimpseScale             (default 1)     
   --glimpseDepth             (default 1)     
   --locatorHiddenSize        (default 128)   
   --imageHiddenSize          (default 512)   
   --wholeImageHiddenSize     (default 256)   

   --pertrain_SR_loss         (default 2)     
   --residual                 (default 1)     
   --rho                      (default 25)    
   --hiddenSize               (default 512)   
   --FastLSTM                 (default 1)     
   --BN                                       
   --save_im                                  
]]

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.ntrain == 'math.huge' then opt.ntrain = math.huge end
opt.loadSize = {128, 128}
opt.gtSize = {256, 256}
opt.hzSize = {256, 256}
local PatchSize = {}
PatchSize[1], PatchSize[2] = opt.glimpsePatchSize:match("([^,]+),([^,]+)")
opt.glimpsePatchSize = {}
opt.glimpsePatchSize[1] = tonumber(PatchSize[1])
opt.glimpsePatchSize[2] = tonumber(PatchSize[2])
opt.glimpseArea = opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2]
if opt.glimpseArea == opt.gtSize[1]*opt.gtSize[2] then
  opt.unitPixels = (opt.gtSize[2] - opt.glimpsePatchSize[2]) / 2
else
  opt.unitPixels = opt.gtSize[2] / 2
end
if opt.display == 0 then opt.display = false end -- lapp argparser cannot handel bool value 

opt.manualSeed = 123 --torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

local DataLoader = paths.dofile('data/data.lua')
-- opt.data = '../train/'
-- local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
opt.data = '../test/'
local dataTest = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') or name:find('Linear') then
      if m.weight then m.weight:normal(0.0, 0.02) end
      if m.bias then m.bias:normal(0.0, 0.02) end
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:normal(0.0, 0.02) end
    end
end

local nc = opt.nc
local rho = opt.rho
local hzSize = opt.hzSize
local gtSize = opt.gtSize

local SpatialBatchNormalization
if opt.BN then SpatialBatchNormalization = nn.SpatialBatchNormalization
else SpatialBatchNormalization = nn.Identity end
local SpatialConvolution = nn.SpatialConvolution

if opt.checkpoints_epoch and opt.checkpoints_epoch > 0 then
  nngraph.annotateNodes()
  print('Loading.. checkpoints_final/' .. opt.checkpoints_name .. '_' .. opt.checkpoints_epoch .. '.t7')
  model = torch.load('checkpoints_final/' .. opt.checkpoints_name .. '_' .. opt.checkpoints_epoch .. '.t7')
else
  local locationSensor = nn.Sequential()
  locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))
  locationSensor:add(nn.BatchNormalization(opt.locatorHiddenSize)):add(nn.ReLU(true))
  local imageSensor = nn.Sequential()
  imageSensor:add(nn.View(-1):setNumInputDims(3))
  imageSensor:add(nn.Linear(nc*gtSize[1]*gtSize[2],opt.wholeImageHiddenSize))
  imageSensor:add(nn.BatchNormalization(opt.wholeImageHiddenSize)):add(nn.ReLU(true))
  local imageErrSensor = nn.Sequential()
  imageErrSensor:add(nn.View(-1):setNumInputDims(3))
  imageErrSensor:add(nn.Linear(nc*gtSize[1]*gtSize[2],opt.wholeImageHiddenSize))
  imageErrSensor:add(nn.BatchNormalization(opt.wholeImageHiddenSize)):add(nn.ReLU(true))
  glimpse = nn.Sequential()
  glimpse:add(nn.ParallelTable():add(locationSensor):add(imageErrSensor):add(imageSensor))
  glimpse:add(nn.JoinTable(1,1))
  glimpse:add(nn.Linear(opt.wholeImageHiddenSize+opt.locatorHiddenSize+opt.wholeImageHiddenSize, opt.imageHiddenSize))
  glimpse:add(nn.BatchNormalization(opt.imageHiddenSize)):add(nn.ReLU(true))
  glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))
  glimpse:add(nn.BatchNormalization(opt.hiddenSize)):add(nn.ReLU(true))
  recurrent = nn.GRU(opt.hiddenSize, opt.hiddenSize)
  local rnn = nn.Recurrent(opt.hiddenSize, glimpse, recurrent, nn.ReLU(true), 99999)
  local locator = nn.Sequential()
  locator:add(nn.Linear(opt.hiddenSize, 2))
  locator:add(nn.Tanh()) 
  locator:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic)) 
  locator:add(nn.HardTanh()) 
  locator:add(nn.MulConstant(opt.unitPixels*2/gtSize[2]))
  
  local i_input_fc = nn.Sequential()
  i_input_fc:add(nn.JoinTable(1,3))
  i_input_fc:add(nn.View(-1):setNumInputDims(3))
  i_input_fc:add(nn.Linear(nc*2*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2],256)):add(nn.ReLU(true))
  i_input_fc:add(nn.Linear(256,256)):add(nn.ReLU(true))
  i_input_fc:add(nn.Linear(256,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
  i_input_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
  local t_img_fc = nn.Sequential()
  t_img_fc:add(nn.JoinTable(1,3))
  t_img_fc:add(nn.View(-1):setNumInputDims(3))
  t_img_fc:add(nn.Linear(nc*2*gtSize[1]*gtSize[2],256)):add(nn.ReLU(true))
  t_img_fc:add(nn.Linear(256,256)):add(nn.ReLU(true))
  t_img_fc:add(nn.Linear(256,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
  t_img_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
  local t_e_fc = nn.Sequential()
  t_e_fc:add(nn.Linear(opt.hiddenSize,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
  t_e_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
  local tEstNet = nn.Sequential()
  tEstNet:add(nn.JoinTable(1,3))
  tEstNet:add(SpatialConvolution(nc*5, 16, 5, 5, 1, 1, 2, 2))
  tEstNet:add(SpatialBatchNormalization(16)):add(nn.ReLU(true))
  tEstNet:add(SpatialConvolution(16, 32, 7, 7, 1, 1, 3, 3))
  tEstNet:add(SpatialBatchNormalization(32)):add(nn.ReLU(true))
  tEstNet:add(SpatialConvolution(32, 64, 7, 7, 1, 1, 3, 3))
  tEstNet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
  tEstNet:add(SpatialConvolution(64, 64, 7, 7, 1, 1, 3, 3))
  tEstNet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
  tEstNet:add(SpatialConvolution(64, 64, 7, 7, 1, 1, 3, 3))
  tEstNet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
  tEstNet:add(SpatialConvolution(64, 32, 7, 7, 1, 1, 3, 3))
  tEstNet:add(SpatialBatchNormalization(32)):add(nn.ReLU(true))
  tEstNet:add(SpatialConvolution(32, 16, 5, 5, 1, 1, 2, 2))
  tEstNet:add(SpatialBatchNormalization(16)):add(nn.ReLU(true))
  tEstNet:add(SpatialConvolution(16, nc, 5, 5, 1, 1, 2, 2))

  local loc_prev = nn.Identity()()
  local image_pre = nn.Identity()()
  local image = nn.Identity()()
  local visited_map_pre = nn.Identity()() -- used for record the attened area
  local onesTensor = nn.Identity()()

  local h = rnn({loc_prev,image_pre,image})
  local loc = locator(h)
  local visited_map = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize)({visited_map_pre, onesTensor, loc})
  local patch = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({image, loc})
  local patch_pre = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({image_pre, loc})
  local i_input_fc_o = i_input_fc({patch, patch_pre})
  local t_img_fc_o = t_img_fc({image, image_pre})
  local t_e_fc_o = t_e_fc(h)
  local t_patch = tEstNet({patch, patch_pre, SR_patch_fc_o, SR_img_fc_o, SR_fc_o})
  if opt.residual then i_patch = nn.Tanh()(nn.CAddTable()({i_patch,patch_pre})) end
  local image_next = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize, nil)({image_pre,i_patch,loc})
  
  nngraph.annotateNodes()
  model = nn.gModule({loc_prev,image_pre,visited_map_pre,onesTensor,image}, {loc, image_next, visited_map})
  model:apply(weights_init)
  -- model.name = 'ddrl'
  model = nn.Recursor(model, opt.rho)
end
gt_glimpse = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)
baseline_R = nn.Sequential()
baseline_R:add(nn.Add(1))
local REINFORCE_Criterion = nn.VRMSEReward(model, opt.rewardScale, opt.rewardAreaScale)
local MSEcriterion = nn.MSECriterion()



optimState = {
learningRate = opt.lr,
beta1 = opt.beta1,
}
local outputs
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   model:cuda()
   baseline_R:cuda()
   MSEcriterion:cuda();      REINFORCE_Criterion:cuda();
   gt_glimpse:cuda()
end
model:forget()
local parameters, gradParameters = model:getParameters()
thin_model = model:sharedClone() 
local a, b = thin_model:getParameters()
print(parameters:nElement())
print(gradParameters:nElement())

testLogger = optim.Logger(paths.concat(opt.name, 'test.log'))
testLogger:setNames{'MSE (training set)', 'PSNR (test set)'}
testLogger.showPlot = false

if opt.display then disp = require 'display' end

local fx = function(x)
  gradParameters:zero()
  model:forget()

  gt, idLabel = data:getBatch()
  hz = gt:clone()
  for imI = 1, gt:size(1) do
    temp = image.scale(gt[imI], hzSize[2], hzSize[1])
    hz[imI] = image.scale(temp, gtSize[2], gtSize[1], 'bicubic')
  end
  gt = gt:cuda()
  hz = hz:cuda()
  idLabel = idLabel:cuda()

  local zero_loc = torch.zeros(opt.batchSize,2)
  local zero_dummy = torch.zeros(opt.batchSize,1)
  local ones = torch.ones(opt.batchSize,1,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2])
  local visited_map0 = torch.zeros(opt.batchSize,1,gtSize[1],gtSize[2])
  zero_loc = zero_loc:cuda()
  zero_dummy = zero_dummy:cuda()
  ones = ones:cuda()
  visited_map0 = visited_map0:cuda()

  local dl = {}
  local inputs = {}
  outputs = {}
  gt = {}
  err_l = 0
  err_g = 0
  
  for t = 1,rho do
    if t == 1 then inputs[t] = {zero_loc, hz, visited_map0, ones, hz}
    else
      inputs[t] = outputs[t-1]
      table.insert(inputs[t], ones)
      table.insert(inputs[t], hz)
    end

    outputs[t] = model:forward(inputs[t])
    gt[t] = gt_glimpse:forward{gt, outputs[t][1]}:clone()

    err_l = err_l + MSEcriterion:forward(outputs[t][2], gt)
    dl[t] = MSEcriterion:backward(outputs[t][2], gt):clone()
  end

  local curbaseline_R = baseline_R:forward(zero_dummy)
  err_g = REINFORCE_Criterion:forward({outputs[rho][2], outputs[rho][3], curbaseline_R}, gt)
  
  local dg = REINFORCE_Criterion:backward({outputs[rho][2], outputs[rho][3], curbaseline_R}, gt)
  
  for t = rho,1,-1 do
    model:backward(inputs[t], {zero_loc, dl[t], visited_map0})
  end

  baseline_R:zeroGradParameters()
  baseline_R:backward(zero_dummy, dg[3])
  baseline_R:updateParameters(0.01)
  return err_g, gradParameters
end

epoch = opt.checkpoints_epoch and opt.checkpoints_epoch or 0
while epoch < opt.niter do
   epoch = epoch+1
   epoch_tm:reset()
   test()
   local counter = 0
   local counter_test = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      collectgarbage()
      tm:reset()

      optim.adam(fx, parameters, optimState)
      a:copy(parameters)
      
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
        local loc_im = torch.Tensor(opt.batchSize,nc,gtSize[1],gtSize[2])
        local p = torch.Tensor(opt.rho,nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2])
        for i = 1,opt.batchSize do
          loc_im[i] = outputs[rho][2][i]:clone():float()
        end
        for t = 1,#gt do
          p[t] = gt[t][1]:clone():float()
        end
        disp.image(loc_im, {win=opt.display_id, title=opt.name..'_output'})
        disp.image(hz, {win=opt.display_id+1, title=opt.name..'_input'})
        disp.image(gt, {win=opt.display_id+2, title=opt.name..'_gt'})
        disp.image(p, {win=opt.display_id+3, title=opt.name..'_gtPatch'})
        disp.image(outputs[rho][3], {win=opt.display_id+4, title=opt.name..'_VisitedMap'})
     end

    end
    paths.mkdir('checkpoints')

    if epoch % opt.epoch == 0 then
      torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_RNN.t7', thin_model)
    end

    -- print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
    --   epoch, opt.niter, epoch_tm:time().real))
end


function test()
  psnr = 0
  model:evaluate()

  paths.mkdir(opt.name)
  for st = 1,dataTest:size(),opt.batchSize do
    model:forget()
    xlua.progress(st,dataTest:size())
    local i2, quantity
    if st + opt.batchSize > dataTest:size() then i2 = dataTest:size() 
    else i2 = st + opt.batchSize - 1 end
    quantity = i2 - st + 1
    gt, impath = dataTest:getIndice({st,i2})
    hz = gt:clone()
    for imI = 1, gt:size(1) do
      temp = image.scale(gt[imI], hzSize[2], hzSize[1])
      hz[imI] = image.scale(temp, gtSize[2], gtSize[1], 'bicubic')
    end
    gt = gt:cuda()
    hz = hz:cuda()

    local zero_loc = torch.zeros(gt:size(1),2):cuda()
    local ones = torch.ones(gt:size(1),1,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):cuda()
    local visited_map0 = torch.zeros(gt:size(1),1,gtSize[1],gtSize[2]):cuda()
    local output_t
    local input_t
    for t = 1,rho do
      if t == 1 then input_t = {zero_loc, hz, visited_map0, ones, hz}
      else
        input_t = {}
        for i = 1, #output_t do input_t[i] = output_t[i]:clone() end
        table.insert(input_t, ones)
        table.insert(input_t, hz)
      end
      output_t = model:forward(input_t)
    end

    for i = 1,quantity do
      psnr = psnr + 10 * math.log10(4 / MSEcriterion:forward(output_t[2][i], gt[i]))
      if opt.save_im then
        local img = output_t[2][i]
        img:add(1):div(2)
        image.save(opt.name..'/'..paths.basename(impath[i]), img)
      end
    end
  end
  psnr = psnr / dataTest:size()
  print(psnr)
  model:training()

  if testLogger then
    paths.mkdir(opt.name)
    testLogger:add{err_g, psnr}
    testLogger:style{'-','-'}
    testLogger:plot()
  end
end