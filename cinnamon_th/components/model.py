import abc
import gc
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Dict, Union, AnyStr, Tuple, Iterator

import numpy as np
from tqdm import tqdm
import torch as th

from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility import logging_utility
from cinnamon_generic.components.callback import Callback, guard
from cinnamon_generic.components.metrics import Metric
from cinnamon_generic.components.model import Network
from cinnamon_generic.components.processor import Processor
from cinnamon_generic.utility.printing_utility import prettify_statistics


class THNetwork(Network):

    # TODO: move to THHelper?
    def get_device(
            self
    ):
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        return device

    def accumulate(
            self,
            accumulator: Dict,
            data: Union[th.Tensor, Dict],
            default_name: str
    ):
        if isinstance(data, th.Tensor):
            accumulator.setdefault(default_name, []).append(data.detach().cpu().numpy())
        else:
            for key, value in data.items():
                accumulator.setdefault(key, []).append(value.detach().cpu().numpy())

        return accumulator

    def input_additional_info(
            self,
            batch_x: Any,
            batch_y: Any
    ) -> Dict:
        return {}

    @abc.abstractmethod
    def batch_loss(
            self,
            batch_x: Any,
            batch_y: Any,
            input_additional_info: Dict = {},
    ) -> Tuple[Any, Any, Dict, Any, Dict]:
        """
        Computes model loss for given batch.

        Args:
            batch_x: batch input data in any model-compliant format
            batch_y: batch ground-truth data in any model-compliant format
            input_additional_info: additional input model data

        Returns:
            The computed loss information for the current step (e.g., losses name and value)
        """

        pass

    def batch_train(
            self,
            batch_x: Any,
            batch_y: Any,
            input_additional_info: Dict = {}
    ) -> Tuple[Any, Any, Dict, Any, Dict]:
        """
        Computes a training step given input data.

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format
            input_additional_info: additional input model data

        Returns:
            The following information is returned:
                - Total loss
                - Total true loss: total loss without any weighting applied (if no weighting is considered, it equals total loss)
                - Loss information: a dictionary with loss name as keys and loss value as values
                - Predictions: model raw output
                - Model additional info: optional intermediate or additional model raw output
        """

        self.optimizer.zero_grad()

        loss, \
            true_loss, \
            loss_info, \
            predictions, \
            model_additional_info = self.batch_loss(batch_x,
                                                    batch_y,
                                                    input_additional_info=input_additional_info)

        loss.backward()

        return loss, true_loss, loss_info, predictions, model_additional_info

    def batch_fit(
            self,
            batch_x: Any,
            batch_y: Any,
            input_additional_info: Dict = {}
    ) -> Tuple[Dict, Any, Dict]:
        """
        Computes a training step given input data.

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format
            input_additional_info: additional input model data

        Returns:
            The computed training information for the current step:
                - Loss information: a dictionary with loss name as keys and loss value as values
                - Predictions: model raw output
                - Model additional info: optional intermediate or additional model raw output
        """

        loss, \
            true_loss, \
            loss_info, \
            predictions, \
            model_additional_info, = self.batch_train(batch_x=batch_x,
                                                      batch_y=batch_y,
                                                      input_additional_info=input_additional_info)

        self.optimizer.step()

        loss_info['loss'] = true_loss
        return loss_info, predictions, model_additional_info

    def batch_predict(
            self,
            batch_x: Any,
            input_additional_info: Dict = {}
    ) -> Tuple[Any, Dict]:
        """
        Computes model predictions for the given input batch.

        Args:
            batch_x: batch input training data in any model-compliant format
            input_additional_info: additional input model data

        Returns:
            - Predictions: model raw output
            - Model additional info: optional intermediate or additional model raw output
        """

        predictions, model_additional_info = self.model(batch_x,
                                                        input_additional_info=input_additional_info,
                                                        training=False)
        return predictions, model_additional_info

    def batch_evaluate(
            self,
            batch_x: Any,
            batch_y: Any,
            input_additional_info: Dict = {}
    ) -> Tuple[Any, Any, Dict, Any, Dict]:
        """
        Computes training loss for the given input batch without a issuing a gradient step.

        Args:
            batch_x: batch input training data in any model-compliant format
            batch_y: batch ground-truth training data in any model-compliant format
            input_additional_info: additional input model data

        Returns:
            The following information is returned:
                - Total loss
                - Total true loss: total loss without any weighting applied (if no weighting is considered, it equals total loss)
                - Loss information: a dictionary with loss name as keys and loss value as values
                - Predictions: model raw output
                - Model additional info: optional intermediate or additional model raw output
        """

        loss, \
            true_loss, \
            loss_info, \
            predictions, \
            model_additional_info = self.batch_loss(batch_x=batch_x,
                                                    batch_y=batch_y,
                                                    input_additional_info=input_additional_info)
        loss_info['loss'] = true_loss
        return loss, true_loss, loss_info, predictions, model_additional_info

    def save_model(
            self,
            filepath: Union[AnyStr, Path]
    ):
        """
        Serializes internal model's weights to filesystem.

        Args:
            filepath: path where to save model's weights.
        """

        if self.model is not None:
            th.save(self.model.state_dict(), filepath.joinpath('weights.pkl'))

    def load_model(
            self,
            filepath: Union[AnyStr, Path]
    ):
        """
        Loads internal model's weights from a serialized checkpoint stored in the filesystem.

        Args:
            filepath: path where the model serialized checkpoint is stored.
        """

        if self.model is not None:
            self.model.load_state_dict(th.load(filepath.joinpath('weights.pkl')))
        return self.model

    @guard()
    def fit(
            self,
            train_data: FieldDict,
            val_data: Optional[FieldDict] = None,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None
    ) -> FieldDict:
        """
        Fits the model with given training and (optionally) validation data.

        Args:
            train_data: training data necessary for training the model
            val_data: validation data that can be used to regularize or monitor the training process
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process
            model_processor: a ``Processor`` component that parses model predictions.

        Returns:
            A ``FieldDict`` storing training information
        """

        logging_utility.logger.info('Training started...')
        logging_utility.logger.info(f'Total steps: {train_data.steps}')

        training_info = {}

        for epoch in range(self.epochs):
            self.model.train()

            if self.model.stop_training:
                logging_utility.logger.info(f'Stopping training at epoch {epoch}')
                break

            if callbacks:
                callbacks.run(hookpoint='on_epoch_begin',
                              logs={'epochs': self.epochs, 'epoch': epoch})

            epoch_info = defaultdict(float)
            batch_idx = 0

            data_iterator: Iterator = train_data.iterator()
            with tqdm(total=train_data.steps, leave=True, position=0, desc=f'Training - Epoch {epoch}') as pbar:
                while batch_idx < train_data.steps:

                    if callbacks:
                        callbacks.run(hookpoint='on_batch_fit_begin',
                                      logs={'batch': batch_idx})

                    batch_x, batch_y = next(data_iterator)
                    input_additional_info = self.input_additional_info(batch_x=batch_x,
                                                                       batch_y=batch_y)
                    batch_info, _, model_additional_info = self.batch_fit(batch_x=batch_x,
                                                                          batch_y=batch_y,
                                                                          input_additional_info=input_additional_info)
                    batch_info = {f'train_{key}': item.detach().cpu().numpy() for key, item in batch_info.items()}

                    if callbacks:
                        callbacks.run(hookpoint='on_batch_fit_end',
                                      logs={'batch': batch_idx,
                                            'batch_info': batch_info,
                                            'model_additional_info': model_additional_info})

                    for key, item in batch_info.items():
                        epoch_info[key] += item

                    batch_idx += 1
                    pbar.update(1)

            epoch_info = {key: item / train_data.steps for key, item in epoch_info.items()}
            epoch_info['epoch'] = epoch + 1

            if val_data is not None:
                val_info = self.evaluate(data=val_data,
                                         callbacks=callbacks,
                                         metrics=metrics,
                                         model_processor=model_processor,
                                         suffixes={'status': 'training', 'split': 'val'})
                val_info = val_info.to_value_dict()

                del val_info['predictions']
                if 'metrics' in val_info:
                    val_info = {**val_info, **val_info['metrics']}
                    del val_info['metrics']

                epoch_info = {**epoch_info, **{f'val_{key}': value for key, value in val_info.items()}}

            logging_utility.logger.info(f'\n{prettify_statistics(epoch_info)}')

            if callbacks:
                callbacks.run(hookpoint='on_epoch_end',
                              logs=epoch_info)

            for key, value in epoch_info.items():
                training_info.setdefault(key, []).append(value)

            # Garbage collect
            gc.collect()

        return FieldDict(training_info)

    @guard()
    def evaluate(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None,
            suffixes: Optional[Dict] = None
    ) -> FieldDict:
        """
        Evaluates a trained model on given data and computes model predictions on the same data.

        Args:
            data: data to evaluate the model on and compute predictions.
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process
            model_processor: a ``Processor`` component that parses model predictions.
            suffixes: suffixes used to uniquely identify evaluation results on input data

        Returns:
            A ``FieldDict`` storing evaluation and prediction information
        """

        loss = defaultdict(float)
        predictions = {}

        data_iterator: Iterator = data.iterator()
        ground_truth = {}

        self.model.eval()

        tqdm_desc = f'Evaluation - {suffixes["split"]}' if suffixes is not None and 'split' in suffixes else 'Evaluation'
        for batch_idx in tqdm(range(data.steps), leave=True, position=0, desc=tqdm_desc):

            if callbacks:
                callbacks.run(hookpoint='on_batch_evaluate_begin',
                              logs={'batch': batch_idx,
                                    'suffixes': suffixes})

            batch_x, batch_y = next(data_iterator)

            ground_truth = self.accumulate(accumulator=ground_truth, data=batch_y, default_name='ground_truth')

            input_additional_info = self.input_additional_info(batch_x=batch_x,
                                                               batch_y=batch_y)
            batch_loss, \
                true_batch_loss, \
                batch_loss_info, \
                batch_predictions, \
                model_additional_info = self.batch_evaluate(batch_x=batch_x,
                                                            batch_y=batch_y,
                                                            input_additional_info=input_additional_info)

            batch_info = {key: item.detach().cpu().numpy() for key, item in batch_loss_info.items()}

            for key, item in batch_info.items():
                loss[key] += item

            if model_processor is not None:
                batch_predictions = model_processor.run(data=batch_predictions)

            predictions = self.accumulate(accumulator=predictions, data=batch_predictions, default_name='predictions')

            if callbacks:
                callbacks.run(hookpoint='on_batch_evaluate_end',
                              logs={'batch': batch_idx,
                                    'batch_info': batch_info,
                                    'batch_loss': batch_loss,
                                    'true_batch_loss': true_batch_loss,
                                    'batch_predictions': batch_predictions,
                                    'batch_y': batch_y,
                                    'model_additional_info': model_additional_info,
                                    'suffixes': suffixes})

        loss = {key: item / data.steps for key, item in loss.items()}

        if 'output_iterator' not in data or metrics is None:
            metrics_info = {}
        else:
            metrics_info = metrics.run(y_pred=predictions, y_true=ground_truth, as_dict=True)

        return FieldDict({**loss, **{'metrics': metrics_info}, **{'predictions': predictions}})

    @guard()
    def predict(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None,
            suffixes: Optional[Dict] = None
    ) -> FieldDict:
        """
        Computes model predictions on the given data.

        Args:
            data: data to compute model predictions.
            callbacks: callbacks for custom execution flow and side effects
            metrics: metrics for quantitatively evaluate the training process
            model_processor: a ``Processor`` component that parses model predictions.
            suffixes: suffixes used to uniquely identify evaluation results on input data

        Returns:
            A ``FieldDict`` storing prediction information
        """

        if 'output_iterator' in data:
            return self.evaluate(data=data,
                                 callbacks=callbacks,
                                 metrics=metrics,
                                 model_processor=model_processor,
                                 suffixes=suffixes)

        predictions = {}

        data_iterator: Iterator = data.input_iterator() if 'input_iterator' in data else data.iterator()

        self.model.eval()

        tqdm_desc = f'Predicting - {suffixes["split"]}' if suffixes is not None and 'split' in suffixes else 'Predicting'
        for batch_idx in tqdm(range(data.steps), leave=True, position=0, desc=tqdm_desc):

            if callbacks:
                callbacks.run(hookpoint='on_batch_predict_begin',
                              logs={'batch': batch_idx,
                                    'suffixes': suffixes})

            batch_x = next(data_iterator)
            input_additional_info = self.input_additional_info(batch_x=batch_x)
            batch_predictions, model_additional_info = self.batch_predict(batch_x=batch_x,
                                                                          input_additional_info=input_additional_info)
            if model_processor is not None:
                batch_predictions = model_processor.run(data=batch_predictions)

            predictions = self.accumulate(accumulator=predictions, data=batch_predictions, default_name='predictions')

            if callbacks:
                callbacks.run(hookpoint='on_batch_predict_end',
                              logs={'batch': batch_idx,
                                    'batch_predictions': batch_predictions,
                                    'model_additional_info': model_additional_info,
                                    'suffixes': suffixes})

        if 'output_iterator' not in data or metrics is None:
            metrics_info = {}
        else:
            ground_truth = {}
            for batch_y in data.output_iterator():
                ground_truth = self.accumulate(accumulator=ground_truth, data=batch_y, default_name='ground_truth')

            metrics_info = metrics.run(y_pred=predictions, y_true=ground_truth, as_dict=True)

        return FieldDict({**{'predictions': predictions}, **{'metrics': metrics_info}})
