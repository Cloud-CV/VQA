require 'nn'
require 'torch'
require 'optim'
require 'misc.DataLoader'
require 'misc.word_level'
require 'misc.phrase_level'
require 'misc.ques_level'
require 'misc.recursive_atten'
require 'misc.cnnModel'
require 'misc.optim_updates'
utils = require 'misc.utils'
require 'xlua'
require 'image'

local TorchModel = torch.class('HieCoattModel')

function TorchModel:__init(vqa_model, cnn_proto, cnn_model, json_file, backend, gpuid)

  self.vqa_model = vqa_model
  self.cnn_proto = cnn_proto
  self.cnn_model = cnn_model
  self.json_file = json_file
  self.backend = backend
  self.gpuid = gpuid

  if self.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if self.backend == 'cudnn' then 
    require 'cudnn' 
    end
    cutorch.setDevice(self.gpuid + 1) -- note +1 because lua is 1-indexed
  end

  self.loadModel()

end


function TorchModel:loadModel()

  local loaded_checkpoint = torch.load(self.vqa_model)
  local lmOpt = loaded_checkpoint.lmOpt

  lmOpt.hidden_size = 512
  lmOpt.feature_type = 'VGG'
  lmOpt.atten_type = 'Alternating'
  cnnOpt = {}
  cnnOpt.cnn_proto = self.cnn_proto
  cnnOpt.cnn_model = self.cnn_model
  cnnOpt.backend = self.backend
  cnnOpt.input_size_image = 512
  cnnOpt.output_size = 512
  cnnOpt.h = 14
  cnnOpt.w = 14
  cnnOpt.layer_num = 37

  -- load the vocabulary and answers.
  local json_file = utils.read_json(self.json_file)
  self.ix_to_word = json_file.ix_to_word
  self.ix_to_ans = json_file.ix_to_ans

  word_to_ix = {}
  for ix, word in pairs(ix_to_word) do
      word_to_ix[word]=ix
  end

  -- load the model
  protos = {}
  protos.word = nn.word_level(lmOpt)
  protos.phrase = nn.phrase_level(lmOpt)
  protos.ques = nn.ques_level(lmOpt)

  protos.atten = nn.recursive_atten()
  protos.crit = nn.CrossEntropyCriterion()
  protos.cnn = nn.cnnModel(cnnOpt)

  if self.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
  end

  cparams, grad_cparams = protos.cnn:getParameters()
  wparams, grad_wparams = protos.word:getParameters()
  pparams, grad_pparams = protos.phrase:getParameters()
  qparams, grad_qparams = protos.ques:getParameters()
  aparams, grad_aparams = protos.atten:getParameters()

  print('Load the weight...')
  wparams:copy(loaded_checkpoint.wparams)
  pparams:copy(loaded_checkpoint.pparams)
  qparams:copy(loaded_checkpoint.qparams)
  aparams:copy(loaded_checkpoint.aparams)

  print('total number of parameters in cnn_model: ', cparams:nElement())
  assert(cparams:nElement() == grad_cparams:nElement())

  print('total number of parameters in word_level: ', wparams:nElement())
  assert(wparams:nElement() == grad_wparams:nElement())

  print('total number of parameters in phrase_level: ', pparams:nElement())
  assert(pparams:nElement() == grad_pparams:nElement())

  print('total number of parameters in ques_level: ', qparams:nElement())
  assert(qparams:nElement() == grad_qparams:nElement())
  protos.ques:shareClones()

  print('total number of parameters in recursive_attention: ', aparams:nElement())
  assert(aparams:nElement() == grad_aparams:nElement())

  self.protos = protos
end



function TorchModel:predict(input_image, question)

  -- load the image
  local img = image.load(input_image)
  -- scale the image
  img = image.scale(img,448,448)
  itorch.image(img)
  img = img:view(1,img:size(1),img:size(2),img:size(3))
  -- parse and encode the question (in a simple way).
  local ques_encode = torch.IntTensor(26):zero()

  local count = 1
  for word in string.gmatch(question, "%S+") do
      ques_encode[count] = self.word_to_ix[word] or self.word_to_ix['UNK']
      count = count + 1
  end
  ques_encode = ques_encode:view(1,ques_encode:size(1))
  -- doing the prediction

  self.protos.word:evaluate()
  self.protos.phrase:evaluate()
  self.protos.ques:evaluate()
  self.protos.atten:evaluate()
  self.protos.cnn:evaluate()

  local image_raw = utils.prepro(img, false)
  image_raw = image_raw:cuda()
  ques_encode = ques_encode:cuda()

  local image_feat = self.protos.cnn:forward(image_raw)
  local ques_len = torch.Tensor(1,1):cuda()
  ques_len[1] = count-1

  local word_feat, img_feat, w_ques, w_img, mask = unpack(self.protos.word:forward({ques_encode, image_feat}))
  local conv_feat, p_ques, p_img = unpack(self.protos.phrase:forward({word_feat, ques_len, img_feat, mask}))
  local q_ques, q_img = unpack(self.protos.ques:forward({conv_feat, ques_len, img_feat, mask}))

  local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img}
  local out_feat = self.protos.atten:forward(feature_ensemble)

  local tmp,pred=torch.max(out_feat,2)
  local ans = self.ix_to_ans[tostring(pred[1][1])]

  print('The answer is: ' .. ans)

  result ={}
  result['input_image'] = input_image
  result['question'] = question
  result['answer'] = answer
  return result

end
