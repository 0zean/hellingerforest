use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{ArrayView1, ArrayView2, Axis, Array2, Array1};
use rayon::prelude::*;
use std::cmp::Ordering;
use itertools::Itertools;
use itertools::Either;
use smallvec::SmallVec;
use std::sync::Arc;
use serde::{Serialize, Deserialize, Serializer};
use smallvec::smallvec;
use rand::prelude::IteratorRandom;

#[derive(Clone, Serialize, Deserialize)]
struct ArcNode(#[serde(with = "arc_node")] Arc<Node>);

mod arc_node {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(node: &Arc<Node>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        node.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<Node>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Node::deserialize(deserializer).map(Arc::new)
    }
}

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
struct Node {
    feature_index: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    value: Option<Vec<f64>>,
}

impl Node {
    fn new() -> Self {
        Node {
            feature_index: None,
            threshold: None,
            left: None,
            right: None,
            value: None,
        }
    }
}

#[pyclass]
#[derive(Clone, Serialize)]
struct RandomForestClassifier {
    #[serde(serialize_with = "serialize_arc_vec")]
    #[serde(deserialize_with = "deserialize_arc_vec")]
    trees: Vec<ArcNode>,
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    n_features: usize,
    n_classes: usize,
}

#[pymethods]
impl RandomForestClassifier {
    #[new]
    fn new(n_estimators: usize, max_depth: usize, min_samples_split: usize) -> Self {
        RandomForestClassifier {
            trees: Vec::new(),
            n_estimators,
            max_depth,
            min_samples_split,
            n_features: 0,
            n_classes: 0,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x = x.as_array();
        let y = y.as_array();
        self.n_features = x.shape()[1];
        self.n_classes = y.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;

        let (x, y) = (Arc::new(x.to_owned()), Arc::new(y.to_owned()));

        self.trees = (0..self.n_estimators)
            .into_par_iter()
            .map(|_| ArcNode(Arc::new(self.build_tree(x.clone(), y.clone(), 0))))
            .collect();

        Ok(())
    }

    fn predict(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
        let x = x.as_array();
        let predictions: Vec<f64> = x
            .outer_iter()
            .map(|sample| self.predict_single(&sample))
            .collect();
        Ok(predictions)
    }
}

impl RandomForestClassifier {
    fn build_tree(&self, x: Arc<Array2<f64>>, y: Arc<Array1<f64>>, depth: usize) -> Node {
        let mut node = Node::new();

        if depth >= self.max_depth || x.nrows() < self.min_samples_split {
            node.value = Some(self.calculate_leaf_value(&y));
            return node;
        }

        let (feature_index, threshold) = self.find_best_split(x.view(), y.view());

        if let (Some(feature_index), Some(threshold)) = (feature_index, threshold) {
            let (left_x, left_y, right_x, right_y) = self.split_data(x.as_ref(), y.as_ref(), feature_index, threshold);

            node.feature_index = Some(feature_index);
            node.threshold = Some(threshold);
            node.left = Some(Box::new(self.build_tree(Arc::new(left_x), Arc::new(left_y), depth + 1)));
            node.right = Some(Box::new(self.build_tree(Arc::new(right_x), Arc::new(right_y), depth + 1)));
        } else {
            node.value = Some(self.calculate_leaf_value(&y));
        }

        node
    }

    fn find_best_split(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> (Option<usize>, Option<f64>) {
        let mut rng = rand::thread_rng();
        let n_features_sqrt = (self.n_features as f64).sqrt().floor() as usize;
        let feature_indices: SmallVec<[usize; 32]> = (0..self.n_features).choose_multiple(&mut rng, n_features_sqrt).into_iter().collect();

        feature_indices.par_iter()
            .map(|&feature_index| {
                let column = x.column(feature_index);
                let mut thresholds: SmallVec<[f64; 32]> = column.iter().cloned().collect();
                thresholds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                thresholds.dedup();

                // Limit the number of threshold candidates
                let n_thresholds = thresholds.len().min(10);
                let step = thresholds.len() / n_thresholds;
                thresholds = thresholds.into_iter().step_by(step).collect();

                thresholds.into_par_iter()
                    .map(|&threshold| {
                        let score = self.calculate_split_score(&x, &y, feature_index, threshold);
                        (feature_index, threshold, score)
                    })
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
            })
            .max_by(|a, b| a.as_ref().unwrap().2.partial_cmp(&b.as_ref().unwrap().2).unwrap_or(Ordering::Equal))
            .and_then(|best| best.map(|(feature_index, threshold, _)| (Some(feature_index), Some(threshold))))
            .unwrap_or((None, None))
    }

    fn calculate_split_score(&self, x: &ArrayView2<f64>, y: &ArrayView1<f64>, feature_index: usize, threshold: f64) -> f64 {
        let column = x.column(feature_index);
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = column
            .iter()
            .enumerate()
            .partition_map(|(i, &value)| {
                if value <= threshold {
                    Either::Left(i)
                } else {
                    Either::Right(i)
                }
            });

        let left_y = y.select(Axis(0), &left_indices);
        let right_y = y.select(Axis(0), &right_indices);

        self.calculate_hellinger_distance(&left_y.view(), &right_y.view())
    }

    fn split_data(&self, x: &Array2<f64>, y: &Array1<f64>, feature_index: usize, threshold: f64) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = x.column(feature_index)
            .iter()
            .enumerate()
            .partition_map(|(i, &value)| {
                if value <= threshold {
                    Either::Left(i)
                } else {
                    Either::Right(i)
                }
            });

        let left_x = x.select(Axis(0), &left_indices);
        let left_y = y.select(Axis(0), &left_indices);
        let right_x = x.select(Axis(0), &right_indices);
        let right_y = y.select(Axis(0), &right_indices);

        (left_x, left_y, right_x, right_y)
    }

    fn calculate_hellinger_distance(&self, left_y: &ArrayView1<f64>, right_y: &ArrayView1<f64>) -> f64 {
        let left_counts = self.count_classes(left_y);
        let right_counts = self.count_classes(right_y);

        let left_total = left_y.len() as f64;
        let right_total = right_y.len() as f64;

        let mut distance = 0.0;
        for i in 0..self.n_classes {
            let left_prob = left_counts[i] as f64 / left_total;
            let right_prob = right_counts[i] as f64 / right_total;
            distance += (left_prob.sqrt() - right_prob.sqrt()).powi(2);
        }

        (distance / 2.0).sqrt()
    }

    fn count_classes(&self, y: &ArrayView1<f64>) -> SmallVec<[usize; 10]> {
        let mut counts = smallvec![0; self.n_classes];
        for &class in y.iter() {
            counts[class as usize] += 1;
        }
        counts
    }

    fn calculate_leaf_value(&self, y: &Arc<Array1<f64>>) -> Vec<f64> {
        let counts = self.count_classes(&y.view());
        counts.into_iter().map(|count| count as f64 / y.len() as f64).collect()
    }

    fn predict_single(&self, sample: &ArrayView1<f64>) -> f64 {
        let predictions: Vec<Vec<f64>> = self.trees
            .iter()
            .map(|tree| self.predict_tree(&tree.0, sample))
            .collect();

        let mut sum_predictions = vec![0.0; self.n_features];
        for pred in predictions {
            for (i, p) in pred.iter().enumerate() {
                sum_predictions[i] += p;
            }
        }

        sum_predictions
            .into_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index as f64)
            .unwrap()
    }

    fn predict_tree(&self, node: &Node, sample: &ArrayView1<f64>) -> Vec<f64> {
        match (node.feature_index, node.threshold) {
            (Some(feature_index), Some(threshold)) => {
                if sample[feature_index] <= threshold {
                    self.predict_tree(node.left.as_ref().unwrap(), sample)
                } else {
                    self.predict_tree(node.right.as_ref().unwrap(), sample)
                }
            }
            _ => node.value.clone().unwrap(),
        }
    }
}

fn serialize_arc_vec<S>(value: &Vec<ArcNode>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    value.serialize(serializer)
}

#[pymodule]
fn hellingerforest(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RandomForestClassifier>()?;
    Ok(())
}
