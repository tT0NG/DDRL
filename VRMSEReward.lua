
local VRMSEReward, parent = torch.class("nn.VRMSEReward", "nn.Criterion")

function VRMSEReward:__init(module, scale, areaScale, criterion)
   parent.__init(self)
   self.module = module 
   self.scale = scale or 1 
   self.areaScale = areaScale or 4 
   self.errorC = nn.MSECriterion()
   self.criterion = criterion or nn.MSECriterion() 
   self.sizeAverage = true
   self.gradInput = {torch.Tensor()}
end

function VRMSEReward:updateOutput(inputTable, target)
   assert(torch.type(inputTable) == 'table')
   local map = inputTable[2]
   local input = inputTable[1]

   -- reward = mse
   self.reward = input:clone()
   self.reward:add(-1, target):pow(2)
   assert(self.reward:dim() == 4)
   for i = 4,2,-1 do
      self.reward = self.reward:sum(i)
   end
   self.reward:resize(self.reward:size(1))
   self.reward:div(-input:size(3)*input:size(4))
   self.reward:add(4) 
   local area = map:sum(4):sum(3):sum(2):div(opt.highResSize[1]*opt.highResSize[2])
   area = area:view(-1)
   self.reward:add(self.areaScale,area) 
   self.output = self.errorC:forward(input,target)
   
   return self.output
end

function VRMSEReward:updateGradInput(inputTable, target)
   local input = inputTable[1]
   local baseline =inputTable[3]
   
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   self.vrReward:add(-1, baseline)
   if self.sizeAverage then
      self.vrReward:div(input:size(1))
   end
   self.module:reinforce(self.vrReward)  
   
   self.gradInput[1]:resizeAs(input):zero()
   self.gradInput[1] = self.gradInput[1]

   self.gradInput[3] = self.criterion:backward(baseline, self.reward)
   self.gradInput[3] = self.gradInput[3]
   return self.gradInput
end

function VRMSEReward:type(type)
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end