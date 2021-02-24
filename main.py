# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import re
import argparse
import os
import shutil
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

import torchvision.transforms as transforms
import torchvision.models as models
from mean_teacher import architectures, datasets, data, losses, ramps, cli ,resnet
from mean_teacher.resnet import resnet50
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.Logger import Logger,savefig,plot_result
from mean_teacher.utils import *
from sklearn.metrics import classification_report,accuracy_score
from tqdm import tqdm

LOG = logging.getLogger('main')

args = None
best_acc = 0
global_step = 0
checkpoint_path = ''

def main(context):
	global global_step
	global best_acc
	global checkpoint_path
	checkpoint_path = context.transient_dir
	print(args.batch_size)
	print(os.path.dirname(__file__))
	training_log = context.create_train_log("training")
	validation_log = context.create_train_log("validation")
	ema_validation_log = context.create_train_log("ema_validation")

	dataset_config = datasets.__dict__[args.dataset]()
	num_classes = dataset_config.pop('num_classes')
		
	train_loader, eval_loader, test_loader, class_names = create_data_loaders(**dataset_config, args=args)

	def create_model(ema=False):
		LOG.info("=> creating {pretrained}{ema} model '{arch}'".format(
			pretrained='pre-trained ' if args.pretrained else '',
			ema='EMA ' if ema else '',
			arch=args.arch))

		# model_factory = architectures.__dict__[args.arch]
		# model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
		# model = model_factory(**model_params)
		model = resnet50(pretrained = True)
		model = nn.DataParallel(model).cuda()

		if ema:
			for param in model.parameters():
				param.detach_()
		return model

	model = create_model()
	ema_model = create_model(ema=True)
	pretrained_resenet = models.resnet50(pretrained = True)

	#load pretrained dict
	pretrained_dict = pretrained_resenet.state_dict()
	model_dict = model.state_dict()
	ema_dict = ema_model.state_dict()
	pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	ema_dict.update(pretrained_dict)

	model.load_state_dict(model_dict)
	ema_model.load_state_dict(ema_dict)
	print('finish loading pretrained model')
	LOG.info(parameters_string(model))

	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay,
								nesterov=args.nesterov)

	# optionally resume from a checkpoint
	if args.resume:
		assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
		LOG.info("=> loading checkpoint '{}'".format(args.resume))
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		global_step = checkpoint['global_step']
		best_acc = checkpoint['best_acc']
		model.load_state_dict(checkpoint['state_dict'])
		ema_model.load_state_dict(checkpoint['ema_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

	cudnn.benchmark = True

	if args.evaluate:
		LOG.info("Evaluating the primary model:")
		validate(eval_loader, model, validation_log, global_step, args.start_epoch,0)
		LOG.info("Evaluating the EMA model:")
		validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch,1)
		return

	logger = Logger(checkpoint_path+'log.txt',title = 'Mean-teacher semi-supervised')
	logger.set_names(['Best_Acc','Student_ACC','EMA_ACC','Validate_Acc','Ema_Valid_Acc'])

	for epoch in range(args.start_epoch, args.epochs):
		start_time = time.time()
		# train for one epoch
		train_loss, train_acc = train(train_loader, model, ema_model, optimizer, epoch, training_log)
		# LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))
		

		#evaluate the performancen
		if args.evaluation_epochs:
			start_time = time.time()
			LOG.info("Evaluating the primary model:")
			prec1 = validate(eval_loader, model, validation_log, global_step, epoch + 1,0)
			LOG.info("Evaluating the EMA model:")
			ema_prec1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1,1)
			LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
			is_best = ema_prec1 > best_acc
			best_acc = max(ema_prec1, best_acc)
		else:
			is_best = False

		if is_best ==True:
			best_epoch = epoch
			save_checkpoint({
				'epoch': epoch + 1,
				'global_step': global_step,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'ema_state_dict': ema_model.state_dict(),
				'best_acc': best_acc,
				'optimizer' : optimizer.state_dict(),
			}, is_best, checkpoint_path, epoch + 1)

		logger.append([best_acc, train_loss, train_acc,prec1,ema_prec1])

	logger.close()
	plot_result(logger,checkpoint_path)
	savefig(os.path.join(checkpoint_path, 'log.eps'))
	best_cp = torch.load(context.transient_dir+'/best_checkpoint.ckpt')['state_dict']
	report,accuracy = evaluate(test_loader,model,best_cp,class_names)
	ema_report,ema_accuracy = evaluate(test_loader,model,best_cp,class_names)
	write_result(checkpoint_path,report,accuracy,ema_report,ema_accuracy,best_epoch)

def write_result(result_path,report,accuracy,ema_report,ema_accuracy,best_epoch):
	writer = open(os.path.join(result_path,'classification_report.txt'),'w')
	writer.write(report+'\n')
	writer.write('best_epoch:'+str(best_epoch)+'\n')
	writer.write('best_acc:'+str(accuracy))

	ema_writer = open(os.path.join(result_path,'ema_classification_report.txt'),'w')
	ema_writer.write(ema_report+'\n')
	ema_writer.write('best_epoch:'+str(best_epoch)+'\n')
	ema_writer.write('best_acc:'+str(ema_accuracy))

def parse_dict_args(**kwargs):
	global args

	def to_cmdline_kwarg(key, value):
		if len(key) == 1:
			key = "-{}".format(key)
		else:
			key = "--{}".format(re.sub(r"_", "-", key))
		value = str(value)
		return key, value

	kwargs_pairs = (to_cmdline_kwarg(key, value)
					for key, value in kwargs.items())
	cmdline_args = list(sum(kwargs_pairs, ()))
	args = parser.parse_args(cmdline_args)


def create_data_loaders(train_transformation,
						eval_transformation,
						test_transformation,
						datadir,
						args):
	traindir = os.path.join(datadir, args.train_subdir)
	evaldir = os.path.join(datadir, args.eval_subdir)
	testdir = os.path.join(datadir, 'test/')

	assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

	dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

	if args.labels: #open the label files: 10% as labels
		root = '/mnt/HDD1/Frederic/Mean-teacher-based'
		label_path  = root+'/data-local/labels/'+args.dataset+'/'+args.labels
		with open(label_path) as f:
			labels = dict(line.split(' ') for line in f.read().splitlines())
		labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)
	if args.exclude_unlabeled:
		sampler = SubsetRandomSampler(labeled_idxs)
		batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
	elif args.labeled_batch_size:
		batch_sampler = data.TwoStreamBatchSampler(
			unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
	else:
		assert False, "labeled batch size {}".format(args.labeled_batch_size)
#dataset.imgs was like {("data-local/1752.jp",-1),("data-local/177.jpg",8)}
	train_loader = torch.utils.data.DataLoader(dataset,
											   batch_sampler=batch_sampler,
											   num_workers=args.workers,
											   pin_memory=True)

	eval_loader = torch.utils.data.DataLoader(
		torchvision.datasets.ImageFolder(evaldir, eval_transformation),
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=2 * args.workers,  # Needs images twice as fast
		pin_memory=True,
		drop_last=False)

	test_loader = torch.utils.data.DataLoader(
		torchvision.datasets.ImageFolder(testdir, test_transformation),
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=2 * args.workers,  # Needs images twice as fast
		pin_memory=True)

	class_names = dataset.classes
	return train_loader, eval_loader, test_loader,class_names


def update_ema_variables(model, ema_model, alpha, global_step):
	# Use the true average until the exponential average is more correct
	alpha = min(1 - 1 / (global_step + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



def train(train_loader, model, ema_model, optimizer, epoch, log):
	global global_step

	class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
	if args.consistency_type == 'mse':
		consistency_criterion = losses.softmax_mse_loss
	elif args.consistency_type == 'kl':
		consistency_criterion = losses.softmax_kl_loss
	else:
		assert False, args.consistency_type
	residual_logit_criterion = losses.symmetric_mse_loss

	meters = AverageMeterSet()

	# switch to train mode
	model.train()
	ema_model.train()

	end = time.time()
	with tqdm(total = len(train_loader)) as pbar:
		# for i, (input, target) in enumerate(train_loader):
		for i, ((input, ema_input), target) in enumerate(train_loader):
			# measure data loading time
			meters.update('data_time', time.time() - end)

			adjust_learning_rate(optimizer, epoch, i, len(train_loader))
			meters.update('lr', optimizer.param_groups [0]['lr'])

			input_var = torch.autograd.Variable(input)
			ema_input_var = torch.autograd.Variable(ema_input, volatile=True)
			target_var = torch.autograd.Variable(target.cuda())

			minibatch_size = len(target_var)
			labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().item()
			assert labeled_minibatch_size > 0
			meters.update('labeled_minibatch_size', labeled_minibatch_size)

			ema_model_out = ema_model(ema_input_var)
			model_out = model(input_var)

			if isinstance(model_out, Variable):
				assert args.logit_distance_cost < 0
				logit1 = model_out
				ema_logit = ema_model_out
			else:
				assert len(model_out) == 2
				assert len(ema_model_out) == 2
				logit1, logit2 = model_out
				ema_logit, _ = ema_model_out

			ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

			if args.logit_distance_cost >= 0:
				class_logit, cons_logit = logit1, logit2
				res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
				meters.update('res_loss', res_loss.item())
			else:
				class_logit, cons_logit = logit1, logit1
				res_loss = 0

			class_loss = class_criterion(class_logit, target_var) / minibatch_size
			meters.update('class_loss', class_loss.item())

			ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
			meters.update('ema_class_loss', ema_class_loss.item())

			if args.consistency:
				consistency_weight = get_current_consistency_weight(epoch)
				meters.update('cons_weight', consistency_weight)
				consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
				meters.update('cons_loss', consistency_loss.item())
			else:
				consistency_loss = 0
				meters.update('cons_loss', 0)

			loss = class_loss + consistency_loss + res_loss
			# loss = class_loss
			assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
			meters.update('loss', loss.item())

			prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1,3))
			meters.update('top1', prec1.item(), labeled_minibatch_size)
			meters.update('error1', 100. - prec1.item(), labeled_minibatch_size)
			meters.update('top5', prec5.item(), labeled_minibatch_size)
			meters.update('error5', 100. - prec5.item(), labeled_minibatch_size)

			ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_var.data, topk=(1, 3))
			meters.update('ema_top1', ema_prec1.item(), labeled_minibatch_size)
			meters.update('ema_error1', 100. - ema_prec1.item(), labeled_minibatch_size)
			meters.update('ema_top5', ema_prec5.item(), labeled_minibatch_size)
			meters.update('ema_error5', 100. - ema_prec5.item(), labeled_minibatch_size)

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			global_step += 1
			update_ema_variables(model, ema_model, args.ema_decay, global_step)

			# measure elapsed time
			meters.update('batch_time', time.time() - end)
			end = time.time()

			# if i % args.print_freq == 0:
			# 	LOG.info(
			# 		'Epoch: [{0}][{1}/{2}]\t'
			# 		'Time {meters[batch_time]:.3f}\t'
			# 		'Data {meters[data_time]:.3f}\t'
			# 		'Class {meters[class_loss]:.4f}\t'
			# 		'Cons {meters[cons_loss]:.4f}\t'
			# 		'Prec@1 {meters[top1]:.3f}\t'
			# 		'Prec@5 {meters[top5]:.3f}'.format(
			# 			epoch, i, len(train_loader), meters=meters))
			pbar.set_description('Epoch%d|%d Acc%.1f ema_Acc%.1f loss%.2f con_l%.2f'%  \
					(epoch,
					args.epochs,
					meters['top1'].average(),
					meters['ema_top1'].average(),
					meters['class_loss'].average(),
					meters['cons_loss'].average()))

			pbar.update(1)

	return meters['top1'].average(),meters['ema_top1'].average()

def validate(eval_loader, model, log, global_step, epoch,ema_sign):
	class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
	meters = AverageMeterSet()

	# switch to evaluate mode
	model.eval()
	pred = []
	targ = []
	end = time.time()
	for i, (input, target) in enumerate(eval_loader):
		meters.update('data_time', time.time() - end)

		input_var = torch.autograd.Variable(input, volatile=True)
		target_var = torch.autograd.Variable(target.cuda(), volatile=True)

		minibatch_size = len(target_var)
		labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().item()
		assert labeled_minibatch_size > 0
		meters.update('labeled_minibatch_size', labeled_minibatch_size)

		# compute output
		output1 = model(input_var)
		# softmax1 = F.softmax(output1, dim=1)
		class_loss = class_criterion(output1[0], target_var) / minibatch_size

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output1[0], target_var.data, topk=(1, 3))
		meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
		meters.update('top1', prec1.item(), labeled_minibatch_size)
		meters.update('error1', 100.0 - prec1.item(), labeled_minibatch_size)
		meters.update('top5', prec5.item(), labeled_minibatch_size)
		meters.update('error5', 100.0 - prec5.item(), labeled_minibatch_size)

		# measure elapsed time
		meters.update('batch_time', time.time() - end)
		end = time.time()

		_,output = torch.max(output1[0],1)
		pred += output.tolist()
		targ += target_var.tolist()

		if i % args.print_freq == 0:
			LOG.info(
				'Test: [{0}/{1}]\t'
				'Time {meters[batch_time]:.3f}\t'
				'Data {meters[data_time]:.3f}\t'
				'Class {meters[class_loss]:.4f}\t'
				'Prec@1 {meters[top1]:.3f}\t'
				'Prec@5 {meters[top5]:.3f}'.format(
					i, len(eval_loader), meters=meters))

	LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
		  .format(top1=meters['top1'], top5=meters['top5']))
	f1 = open(checkpoint_path+'targ.txt','w')
	f1.write(str(targ))
	if ema_sign == 0:
		f = open(checkpoint_path+'pred.txt','w')
		f.write(str(pred))
	elif ema_sign == 1:
		f = open(checkpoint_path+'ema_pred.txt','w')
		f.write(str(pred))

	return meters['top1'].avg

def evaluate(test_loader, model,checkpoint,class_names):
	class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
	# switch to evaluate mode
	model.load_state_dict(checkpoint)
	model.eval()
	pred = []
	targ = []
	end = time.time()
	for i, (input, target) in enumerate(test_loader):

		input_var = torch.autograd.Variable(input, volatile=True)
		target_var = torch.autograd.Variable(target.cuda(), volatile=True)

		outputs = model(input_var)
		_,output = torch.max(outputs[0],1)
		pred += output.tolist()
		targ += target_var.tolist()

	report = classification_report(targ,pred,digits = 4,target_names =class_names)
	accuracy = accuracy_score(targ,pred)
	return report,accuracy

def save_checkpoint(state, is_best, dirpath, epoch):
	filename = 'best_checkpoint.ckpt'
	checkpoint_path = os.path.join(dirpath, filename)
	torch.save(state, checkpoint_path)
	LOG.info("---best checkpoint saved to %s ---" % checkpoint_path)	


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
	lr = args.lr
	epoch = epoch + step_in_epoch / total_steps_in_epoch

	# LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
	lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

	# Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
	if args.lr_rampdown_epochs:
		assert args.lr_rampdown_epochs >= args.epochs
		lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def get_current_consistency_weight(epoch):
	# Consistency ramp-up from https://arxiv.org/abs/1610.02242
	return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
	# return args.consistency

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
	return res

# def create_normal_data_loaders(path):
#     #Load data and augment train data
#     data_transforms = {
#         #Augment the trainning data
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224), #crop the given image
#             transforms.RandomHorizontalFlip(),  #horizontally flip the image
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         #Scale and normalize the validation data
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         #Scale and normalize the validation data
#         'test': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
# }

#     data_dir = path
#     # testData_dir = '/mnt/HDD1/Frederic/ensemble_baseline/TestImage/'

#     image_datasets = {
#             x : torchvision.datasets.ImageFolder(os.path.join(data_dir,x),
#                                  data_transforms[x])
#             for x in ['train','val','test']
#         }

#     train_loader = torch.utils.data.DataLoader(image_datasets['train'],     
#                                                         batch_size=args.batch_size, 
#                                                         shuffle=True,
#                                                         num_workers=0) 
													
#     eval_loader = torch.utils.data.DataLoader(image_datasets['val'],     
#                                                         batch_size=args.batch_size, 
#                                                         shuffle=True,
#                                                         num_workers=0)                                                 

#     testImageLoader = torch.utils.data.DataLoader(image_datasets['test'],batch_size=16,shuffle=False)

#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
#     class_names = image_datasets['train'].classes
#     numOfClasses = len(class_names)

#     return train_loader,eval_loader,testImageLoader,class_names


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	args = cli.parse_commandline_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
	main(RunContext(__file__,args.consistency,args.epochs,args.labels))
