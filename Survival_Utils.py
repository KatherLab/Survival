# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:01:18 2021

@author: Narmin Ghaffari Laleh
"""

import os 
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch.optim as optim
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import gc
import tensorflow as tf
from typing import Dict, Iterable, Sequence, Tuple, Optional
from sksurv.metrics import concordance_index_censored
import tensorflow.compat.v2.summary as summary
from tensorflow.python.ops import summary_ops_v2
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import glob

##############################################################################

def CreateCleanedTable(imagesPath, cliniTablePath, slideTablePath, targetNames, outputPath):
    
    if cliniTablePath.split('.')[-1] == 'csv':
        clinicalTable = pd.read_csv(cliniTablePath)
    else:
        clinicalTable = pd.read_excel(cliniTablePath)
    
    if slideTablePath.split('.')[-1] == 'csv':
        slideTable = pd.read_csv(slideTablePath)
    else:
        slideTable = pd.read_excel(slideTablePath)
        
        
    clinicalTable_Patient = list(clinicalTable['PATIENT'])
    clinicalTable_Patient = list(set(clinicalTable_Patient))
    clinicalTable_Patient = [i for i in clinicalTable_Patient if not str(i) == 'nan']
    
    slideTable_Patint = list(slideTable['PATIENT'])
    slideTable_PatintNotUnique = slideTable_Patint.copy()
    slideTable_Patint = list(set(slideTable_Patint))
    slideTable_Patint = [i for i in slideTable_Patint if not str(i) == 'nan']
    
    inClinicalNotInSlide = []
    for item in clinicalTable_Patient:
        if not item in slideTable_Patint:
            inClinicalNotInSlide.append(item)
    print('\n')
    print('Could not find ' + str(len(inClinicalNotInSlide)) + ' Patients from Clini Table in Slide Table! \n')
    print(inClinicalNotInSlide)
    print('**********************************************************************')  
    
    
    inSlideNotInClinical = []
    for item in slideTable_Patint:
        if not item in clinicalTable_Patient:
            inSlideNotInClinical.append(item)    
    print('Could not find ' + str(len(inSlideNotInClinical)) + ' Patients from Slide Table in Clini Table! \n')               
    print(inSlideNotInClinical)
    print('**********************************************************************') 
    
    
    patient_Diff  = list(set(clinicalTable_Patient) ^ set(slideTable_Patint))

    if len(patient_Diff):
        print('**********************************************************************')
        print('The Slide Table  has: ' + str(len(slideTable_Patint)) + ' patients')
        print('The Clinical Table  has: ' + str(len(clinicalTable_Patient)) + ' patients')
        print('There are difference of: ' + str(len(patient_Diff)))
        print('**********************************************************************')
        
    patienID_temp = []
    for item in clinicalTable_Patient:
        if item in slideTable_Patint:
            patienID_temp.append(item)        
    patientIDs = []
    for item in patienID_temp:
        if item in clinicalTable_Patient:
            patientIDs.append(item)
    
    patientIDs = list(set(patientIDs))    
    
    ids = []
    matSlides = []
    imgsAdress = []
    targets = {}
    for targetName in targetNames:
        targets[targetName] = []
        
    for patientID in patientIDs:
        indicies = [i for i, n in enumerate(slideTable_PatintNotUnique) if n == patientID]
        matchedSlides = [list(slideTable['FILENAME'])[i] for i in indicies] 
        
        temp = clinicalTable.loc[clinicalTable['PATIENT'] == patientID]
        temp.reset_index(drop=True, inplace=True)
        for slide in matchedSlides:
            #globName = glob.glob(os.path.join(imagesPath, str(slide)) + '.*')
            globName = glob.glob(os.path.join(imagesPath, str(slide)))
            if not len(globName) == 0:
                if os.path.exists(globName[0]):  
                    ids.append(patientID)
                    matSlides.append(slide)
                    imgsAdress.append(os.path.join(imagesPath, str(slide)))          
                    for targetName in targetNames:
                        targets[targetName].append(temp[targetName][0])
                    break
            
    data = pd.DataFrame()
    data['imgsAdress'] = imgsAdress
    data['PATIENT'] = ids
    data['FILENAME'] = matSlides
    for targetName in targetNames:
        data[targetName] = targets[targetName]
    data.to_csv(os.path.join(outputPath, 'Cleaned_Data.csv'),  index = False)
    return data, os.path.join(outputPath, 'Cleaned_Data.csv')

###############################################################################

def GetTiles(patients, time, event, imgsList, cleanedTable, maxBlockNum = 500, test = False):

    slideTable_PatintNotUnique = list(cleanedTable['PATIENT'])    
    imgsList = [os.path.join(imgsList, i) for i in os.listdir(imgsList)]
    tilesPathList = []
    timeList = []
    eventList = []
    patinetList = []
    for index, patientID in enumerate(tqdm(patients)):
        indicies = [i for i, n in enumerate(slideTable_PatintNotUnique) if n == patientID]
        matchedSlides = [list(cleanedTable['FILENAME'])[i] for i in indicies] 
    
        for slide in matchedSlides:
            slide= str(slide)
            slide = slide.replace(' ', '')
            sld = [it for it in imgsList if str(slide) in it]
            if not len(sld) == 0:
                slideAdress = sld[0]
                slideContent = os.listdir(slideAdress)
                if len(slideContent) > maxBlockNum:
                    slideContent = np.random.choice(slideContent, maxBlockNum, replace=False)
                for tile in slideContent:
                    tileAdress = os.path.join(slideAdress, tile)                    
                    tilesPathList.append(tileAdress)
                    timeList.append(time[index])
                    eventList.append(event[index])
                    patinetList.append(str(patientID))
 
    # WRITE THEM TO THE EXCEL FILES:
    df = pd.DataFrame(list(zip(patinetList, tilesPathList, timeList, eventList)), columns =['PATIENT', 'TILEPATH', 'TIME', 'EVENT']) 
    
    df_temp = df.dropna()
    if test:
        dfFromDict = df_temp
    else:            
        tags = list(df_temp['EVENT'].unique())
        tagsLength = []
        dfs = {}
        for tag in tags:
            temp = df_temp.loc[df_temp['EVENT'] == tag]
            temp = temp.sample(frac=1).reset_index(drop=True)
            dfs[tag] = temp 
            tagsLength.append(len(df_temp.loc[df_temp['EVENT'] == tag]))
        
        minSize = np.min(tagsLength)
        keys = list(dfs.keys())
        frames = []
        for key in keys:
            temp_len = len(dfs[key])
            diff_len = temp_len - minSize
            drop_indices = np.random.choice(dfs[key].index, diff_len, replace = False)
            frames.append(dfs[key].drop(drop_indices))
            
        dfFromDict = pd.concat(frames)
                    
    return dfFromDict


###############################################################################

def Set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
###############################################################################

class DatasetLoader(torch.utils.data.Dataset):

    def __init__(self, imgs, times, events, transform = None, target_patch_size = -1):
        'Initialization'
        self.times = times
        self.events = events
        self.imgs = imgs
        self.target_patch_size = target_patch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = Image.open(self.imgs[index])
        time = self.times[index]
        event = self.events[index]
        if self.target_patch_size is  not None:
            X = X.resize((self.target_patch_size, self.target_patch_size))
            X = np.array(X)
        if self.transform is not None:
            X = self.transform(X)
        return X, time, event
    
###############################################################################

def get_optim(model, args, params = False):
   
    if params:
        temp = model
    else:
        temp = filter(lambda p: p.requires_grad, model.parameters())
        
    if args.opt == "adam":
        optimizer = optim.Adam(temp, lr = args.lr, weight_decay = args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(temp, lr = args.lr, momentum = 0.9, weight_decay = args.reg)
    else:
        raise NotImplementedError
        
    return optimizer

###############################################################################     
  
class Predictor:

    def __init__(self, model, model_dir):
        self.model = model
        self.model_dir = model_dir

    def predict(self, dataset):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            optimizer=tf.keras.optimizers.Adam(),
            model=self.model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, str(self.model_dir), max_to_keep=2)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            #print(f"Latest checkpoint restored from {ckpt_manager.latest_checkpoint}.")

        risk_scores = []
        #for batch in dataset:
        #print(batch)
        pred = self.model(dataset, training=False)
        risk_scores.append(pred.numpy())    

        return np.row_stack(risk_scores)
    
###############################################################################

def _make_riskset(time: np.ndarray) -> np.ndarray:
    """Compute mask that represents each sample's risk set.

    Parameters
    ----------
    time : np.ndarray, shape=(n_samples,)
        Observed event time sorted in descending order.

    Returns
    -------
    risk_set : np.ndarray, shape=(n_samples, n_samples)
        Boolean matrix where the `i`-th row denotes the
        risk set of the `i`-th instance, i.e. the indices `j`
        for which the observer time `y_j >= y_i`.
    """
    assert time.ndim == 1, "expected 1D array"

    # sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set

###############################################################################

class InputFunction:
    """Callable input function that computes the risk set for each batch.
    
    Parameters
    ----------
    images : np.ndarray, shape=(n_samples, height, width)
        Image data.
    time : np.ndarray, shape=(n_samples,)
        Observed time.
    event : np.ndarray, shape=(n_samples,)
        Event indicator.
    batch_size : int, optional, default=64
        Number of samples per batch.
    drop_last : int, optional, default=False
        Whether to drop the last incomplete batch.
    shuffle : bool, optional, default=False
        Whether to shuffle data.
    seed : int, optional, default=89
        Random number seed.
    """

    def __init__(self,
                 images: np.ndarray,
                 time: np.ndarray,
                 event: np.ndarray,
                 batch_size: int = 10,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 seed: int = 89) -> None:
        if images.ndim == 3:
            images = images[..., np.newaxis]
        self.images = images
        self.time = time
        self.event = event
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

    def size(self) -> int:
        """Total number of samples."""
        return self.images.shape[0]

    def steps_per_epoch(self) -> int:
        """Number of batches for one epoch."""
        return int(np.floor(self.size() / self.batch_size))

    def _get_data_batch(self, index: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute risk set for samples in batch."""
        time = self.time[index]
        event = self.event[index]
        images = self.images[index]

        labels = {
            "label_event": event.astype(np.int32),
            "label_time": time.astype(np.float32),
            "label_riskset": _make_riskset(time)
        }
        return images, labels

    def _iter_data(self) -> Iterable[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Generator that yields one batch at a time."""
        index = np.arange(self.size())
        rnd = np.random.RandomState(self.seed)

        if self.shuffle:
            rnd.shuffle(index)
        for b in range(self.steps_per_epoch()):
            start = b * self.batch_size
            idx = index[start:(start + self.batch_size)]
            yield self._get_data_batch(idx)

        if not self.drop_last:
            start = self.steps_per_epoch() * self.batch_size
            idx = index[start:]
            yield self._get_data_batch(idx)

    def _get_shapes(self) -> Tuple[tf.TensorShape, Dict[str, tf.TensorShape]]:
        """Return shapes of data returned by `self._iter_data`."""
        batch_size = self.batch_size if self.drop_last else None
        h, w, c = self.images.shape[1:]
        images = tf.TensorShape([batch_size, h, w, c])

        labels = {k: tf.TensorShape((batch_size,))
                  for k in ("label_event", "label_time")}
        labels["label_riskset"] = tf.TensorShape((batch_size, batch_size))
        return images, labels

    def _get_dtypes(self) -> Tuple[tf.DType, Dict[str, tf.DType]]:
        """Return dtypes of data returned by `self._iter_data`."""
        labels = {"label_event": tf.int32,
                  "label_time": tf.float32,
                  "label_riskset": tf.bool}
        return tf.float32, labels

    def _make_dataset(self) -> tf.data.Dataset:
        """Create dataset from generator."""
        ds = tf.data.Dataset.from_generator(
            self._iter_data,
            self._get_dtypes(),
            self._get_shapes()
        )
        return ds

    def __call__(self) -> tf.data.Dataset:
        return self._make_dataset()

###############################################################################

def safe_normalize(x: tf.Tensor) -> tf.Tensor:
    """Normalize risk scores to avoid exp underflowing.

    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min = tf.reduce_min(x, axis=0)
    c = tf.zeros_like(x_min)
    norm = tf.where(x_min < 0, -x_min, c)
    return x + norm

###############################################################################

def logsumexp_masked(risk_scores: tf.Tensor,
                     mask: tf.Tensor,
                     axis: int = 0,
                     keepdims: Optional[bool] = None) -> tf.Tensor:
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    risk_scores.shape.assert_same_rank(mask.shape)

    with tf.name_scope("logsumexp_masked"):
        mask_f = tf.cast(mask, risk_scores.dtype)
        risk_scores_masked = tf.math.multiply(risk_scores, mask_f)
        # for numerical stability, substract the maximum value
        # before taking the exponential
        amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
        risk_scores_shift = risk_scores_masked - amax

        exp_masked = tf.math.multiply(tf.exp(risk_scores_shift), mask_f)
        exp_sum = tf.reduce_sum(exp_masked, axis=axis, keepdims=True)
        output = amax + tf.math.log(exp_sum)
        if not keepdims:
            output = tf.squeeze(output, axis=axis)
    return output

###############################################################################

class CoxPHLoss(tf.keras.losses.Loss):
    """Negative partial log-likelihood of Cox's proportional hazards model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)            

    def call(self,
             y_true: Sequence[tf.Tensor],
             y_pred: tf.Tensor) -> tf.Tensor:
        """Compute loss.

        Parameters
        ----------
        y_true : list|tuple of tf.Tensor
            The first element holds a binary vector where 1
            indicates an event 0 censoring.
            The second element holds the riskset, a
            boolean matrix where the `i`-th row denotes the
            risk set of the `i`-th instance, i.e. the indices `j`
            for which the observer time `y_j >= y_i`.
            Both must be rank 2 tensors.
        y_pred : tf.Tensor
            The predicted outputs. Must be a rank 2 tensor.

        Returns
        -------
        loss : tf.Tensor
            Loss for each instance in the batch.
        """
        event, riskset = y_true
        predictions = y_pred

        pred_shape = predictions.shape
        if pred_shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "be 2." % pred_shape.ndims)

        if pred_shape[1] is None:
            raise ValueError("Last dimension of predictions must be known.")

        if pred_shape[1] != 1:
            raise ValueError("Dimension mismatch: Last dimension of predictions "
                             "(received %s) must be 1." % pred_shape[1])

        if event.shape.ndims != pred_shape.ndims:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "equal rank of event (received %s)" % (
                pred_shape.ndims, event.shape.ndims))

        if riskset.shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of riskset (received %s) should "
                             "be 2." % riskset.shape.ndims)

        event = tf.cast(event, predictions.dtype)
        predictions = safe_normalize(predictions)
        print(predictions)

        with tf.name_scope("assertions"):
            assertions = (
                tf.debugging.assert_less_equal(event, 1.),
                tf.debugging.assert_greater_equal(event, 0.),
                tf.debugging.assert_type(riskset, tf.bool)
            )

        # move batch dimension to the end so predictions get broadcast
        # row-wise when multiplying by riskset
        pred_t = tf.transpose(predictions)
        # compute log of sum over risk set for each row
        rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)
        assert rr.shape.as_list() == predictions.shape.as_list()

        losses = tf.math.multiply(event, rr - predictions)

        return losses

###############################################################################

class CindexMetric:
    """Computes concordance index across one epoch."""

    def reset_states(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {
            "label_time": [],
            "label_event": [],
            "prediction": []
        }

    def update_state(self, y_true: Dict[str, tf.Tensor], y_pred: tf.Tensor) -> None:
        """Collect observed time, event indicator and predictions for a batch.

        Parameters
        ----------
        y_true : dict
            Must have two items:
            `label_time`, a tensor containing observed time for one batch,
            and `label_event`, a tensor containing event indicator for one batch.
        y_pred : tf.Tensor
            Tensor containing predicted risk score for one batch.
        """
        self._data["label_time"].append(y_true["label_time"].numpy())
        self._data["label_event"].append(y_true["label_event"].numpy())
        self._data["prediction"].append(tf.squeeze(y_pred).numpy())

    def result(self) -> Dict[str, float]:
        """Computes the concordance index across collected values.

        Returns
        ----------
        metrics : dict
            Computed metrics.
        """
        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)

        results = concordance_index_censored(
            data["label_event"] == 1,
            data["label_time"],
            data["prediction"])

        result_data = {}
        names = ("cindex", "concordant", "discordant", "tied_risk")
        for k, v in zip(names, results):
            result_data[k] = v

        return result_data

###############################################################################

class TrainAndEvaluateModel:

    def __init__(self, model, model_dir, train_dataset, eval_dataset,
                 learning_rate, num_epochs):
        self.num_epochs = num_epochs
        self.model_dir = model_dir

        self.model = model

        self.train_ds = train_dataset
        self.val_ds = eval_dataset

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = CoxPHLoss()

        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        self.val_cindex_metric = CindexMetric()

    @tf.function
    def train_one_step(self, x, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        with tf.GradientTape() as tape:
            logits = self.model(x, training = True)

            train_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=logits)

        with tf.name_scope("gradients"):
            grads = tape.gradient(train_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return train_loss, logits

    def train_and_evaluate(self):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            optimizer=self.optimizer,
            model=self.model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, str(self.model_dir), max_to_keep=2)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f"Latest checkpoint restored from {ckpt_manager.latest_checkpoint}.")

        train_summary_writer = summary.create_file_writer(str(self.model_dir / "train"))
        if self.val_ds:
            val_summary_writer = summary.create_file_writer(str(self.model_dir / "valid"))

        for epoch in range(self.num_epochs):
            gc.collect()
            print('Epoch = ' + str(epoch))
            with train_summary_writer.as_default():
                print('ckpt.step' + str(ckpt.step))
                self.train_one_epoch(ckpt.step)

            # Run a validation loop at the end of each epoch.
            if self.val_ds:
                with val_summary_writer.as_default():
                    val_logits = self.evaluate(ckpt.step)
            else:
                val_logits = []

        save_path = ckpt_manager.save()
        print(f"Saved checkpoint for step {ckpt.step.numpy()}: {save_path}")
        return val_logits

    def train_one_epoch(self, step_counter):
        for x, y in tqdm(self.train_ds):
            train_loss, logits = self.train_one_step(x, y["label_event"], y["label_riskset"])

            step = int(step_counter)
            if step == 0:
                # see https://stackoverflow.com/questions/58843269/display-graph-using-tensorflow-v2-0-in-tensorboard
                func = self.train_one_step.get_concrete_function(x, y["label_event"], y["label_riskset"])
                #summary_ops_v2.graph(func.graph, step=0)

            # Update training metric.
            self.train_loss_metric.update_state(train_loss)

            # Log every 200 batches.
            if step % 100 == 0:
                # Display metrics
                mean_loss = self.train_loss_metric.result()
                print(f"step {step}: mean loss = {mean_loss:.4f}")
                # save summaries
                summary.scalar("loss", mean_loss, step=step_counter)
                # Reset training metrics
                self.train_loss_metric.reset_states()

            step_counter.assign_add(1)

    @tf.function
    def evaluate_one_step(self, x, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        val_logits = self.model(x, training=False)
        val_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=val_logits)
        return val_loss, val_logits

    def evaluate(self, step_counter):
        valLogitList = []
        self.val_cindex_metric.reset_states()
        
        for x_val, y_val in tqdm(self.val_ds):
            val_loss, val_logits = self.evaluate_one_step(
                x_val, y_val["label_event"], y_val["label_riskset"])
            valLogitList.append(val_logits)
            # Update val metrics
            self.val_loss_metric.update_state(val_loss)
            self.val_cindex_metric.update_state(y_val, val_logits)

        val_loss = self.val_loss_metric.result()
        summary.scalar("loss",
                       val_loss,
                       step=step_counter)
        self.val_loss_metric.reset_states()
        
        try:
            val_cindex = self.val_cindex_metric.result()
            for key, value in val_cindex.items():
                summary.scalar(key, value, step=step_counter)
    
            print(f"Validation: loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}")
        except:
            print('All samples are censored')
        return valLogitList
    
###############################################################################

def MergeResults(resultFolderPath):
    
    files = [os.path.join(resultFolderPath , i) for i in os.listdir(resultFolderPath)]
    files = [i for i in files if not 'TILE_BASED' in i]
    df = pd.DataFrame()
    for file in files:
        df = df.append(pd.read_csv(file), ignore_index=True) 
    df.to_csv(os.path.join(resultFolderPath, 'TOTAL_TEST_RESULTS_PATIENT_BASED.csv'))
    
    
    cindex = concordance_index_censored(
      event_indicator = df['EVENT'] == 1,
      event_time = df['TIME'],
      estimate = df['SCORE'])
    return cindex

###############################################################################

def CalculateCIndex(patientBaseResultPath):
    df = pd.read_csv(patientBaseResultPath)
    cindex = concordance_index_censored(
      event_indicator = df['EVENT'] == 1,
      event_time = df['TIME'],
      estimate = df['SCORE'])
    return cindex

###############################################################################












































