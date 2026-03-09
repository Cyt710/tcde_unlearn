import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip_train.precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def run_cifar_forget_eval(model, dataloader, args, forget_class_id=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    cifar100_classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

    logging.info("CIFAR-100 zero-shot Classification")
    tokenizer = get_tokenizer(args.model)
    
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=cifar100_classes,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=5, 
            device=device,
            use_tqdm=True,
        )

    results = {}
    
    try:
        with torch.inference_mode():
            forget_top1, forget_top5, forget_n = 0., 0., 0.
            retain_top1, retain_top5, retain_n = 0., 0., 0.
            all_top1, all_top5, all_n = 0., 0., 0.

            for i, batch in enumerate(tqdm(dataloader, unit_scale=args.batch_size)):
                from open_clip_train.train import prepare_batch_data
                
                try:
                    images, texts, is_forget, labels = prepare_batch_data(batch, device, input_dtype)
                    targets = labels
                    
                    images = images.to(device=device, dtype=input_dtype)
                    targets = targets.to(device)

                    with autocast():
                        output = model(image=images)
                        image_features = output['image_features'] if isinstance(output, dict) else output[0]
                        logits = 100. * image_features @ classifier

                    acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                    all_top1 += acc1
                    all_top5 += acc5
                    all_n += images.size(0)

                    if is_forget is not None and torch.is_tensor(is_forget):
                        forget_mask = is_forget.bool().to(device)
                        retain_mask = ~forget_mask

                        if forget_mask.any():
                            forget_acc1, forget_acc5 = accuracy(logits[forget_mask], targets[forget_mask], topk=(1, 5))
                            forget_top1 += forget_acc1
                            forget_top5 += forget_acc5
                            forget_n += forget_mask.sum().item()

                        if retain_mask.any():
                            retain_acc1, retain_acc5 = accuracy(logits[retain_mask], targets[retain_mask], topk=(1, 5))
                            retain_top1 += retain_acc1
                            retain_top5 += retain_acc5
                            retain_n += retain_mask.sum().item()

                    if (i + 1) % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logging.warning(f"batch {i} failure: {e}")
                    continue

        if all_n > 0:
            results['cifar_all_top1'] = all_top1 / all_n
            results['cifar_all_top5'] = all_top5 / all_n
        
        if forget_n > 0:
            results['cifar_forget_top1'] = forget_top1 / forget_n
            results['cifar_forget_top5'] = forget_top5 / forget_n
        
        if retain_n > 0:
            results['cifar_retain_top1'] = retain_top1 / retain_n
            results['cifar_retain_top5'] = retain_top5 / retain_n

        if 'cifar_forget_top1' in results and 'cifar_retain_top1' in results:
            results['cifar_forgetting_effectiveness'] = results['cifar_retain_top1'] - results['cifar_forget_top1']
            
    except Exception as e:
        logging.error(f"CIFAR evaluation failure: {e}")
        
        return {}

    return results


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
        
    if args.distributed and not args.horovod:
        model = model.module

    logging.info(f'Starting zero-shot evaluation for epoch {epoch}')
    
    results = {}

    if 'imagenet-val' in data or 'imagenet-v2' in data:
        try:
            if tokenizer is None:
                tokenizer = get_tokenizer(args.model)

            logging.info('Building zero-shot classifier for ImageNet')
            device = torch.device(args.device)
            autocast = get_autocast(args.precision, device_type=device.type)
            
            with autocast():
                classifier = build_zero_shot_classifier(
                    model,
                    tokenizer=tokenizer,
                    classnames=IMAGENET_CLASSNAMES,
                    templates=OPENAI_IMAGENET_TEMPLATES,
                    num_classes_per_batch=5,  
                    device=device,
                    use_tqdm=True,
                )

            logging.info('Running ImageNet evaluation')
            if 'imagenet-val' in data:
                top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
                results['imagenet-zeroshot-val-top1'] = top1
                results['imagenet-zeroshot-val-top5'] = top5
                
            if 'imagenet-v2' in data:
                top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
                results['imagenetv2-zeroshot-val-top1'] = top1
                results['imagenetv2-zeroshot-val-top5'] = top5
                
            logging.info('Finished ImageNet zero-shot evaluation')
            
        except Exception as e:
            logging.error(f"ImageNet zero-shot evaluation failed: {e}")

    if 'val' in data and hasattr(args, 'dataset_type') and 'cifar' in args.dataset_type.lower():
        logging.info('Starting simplified CIFAR forget-set evaluation')
        try:
            cifar_results = run_cifar_forget_eval(model, data['val'].dataloader, args)
            results.update(cifar_results)
            
            if cifar_results:
                logging.info('CIFAR Forget-Set Evaluation Results:')
                for key, value in cifar_results.items():
                    if isinstance(value, float):
                        logging.info(f"  {key}: {value:.4f}")
                        
        except Exception as e:
            logging.error(f"CIFAR forget-set evaluation failed: {e}")
        
        logging.info('Finished CIFAR forget-set evaluation')

    logging.info(f'Completed zero-shot evaluation for epoch {epoch}')
    return results