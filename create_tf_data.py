import numpy as np
import tensorflow as tf
from pyteomics import mgf
import pandas as pd
from scipy.sparse import csr_matrix

def preprocess_spectrum(ID,mz,intensity):
    MZ_MAX = 2000
    SPECTRUM_RESOLUTION = 0

    def _parse_indices(element):
        resolution = SPECTRUM_RESOLUTION
        element = round(element,resolution)
        element = int(element * (10**resolution))
        return [element]

    def _rescale_spectrum(indices,values):
        # get unique indices, and positions in the array
        y,idx = np.unique(indices,return_index=True)
        
        # Use the positions of the unique values as the segment ids to sum segments up:
        values = np.add.reduceat(values, idx)
        
        ## Truncate
        mask = np.less(y,MZ_MAX * (10**SPECTRUM_RESOLUTION))
        indices = y[mask]
        values = values[mask]
        
        #make nested list [1, 2, 3] -> [[1],[2],[3]], as later requiered by SparseTensor:
        #indices = tf.reshape(indices, [tf.size(indices),1])        

        return indices,values
    
    def _to_sparse(indices,values):
        from scipy.sparse import csr_matrix        
        zeros = np.zeros(len(indices),dtype=np.int32)        
        intensities_array = csr_matrix((values,(indices,zeros)),shape=(MZ_MAX * (10**SPECTRUM_RESOLUTION),1), dtype=np.float64).toarray().flatten()
        return intensities_array


    #### PREPROCESSING BEGIN #######
    # round indices according to spectrum resolution, SPECTRUM_RESOLUTION:
    mz = list(map(_parse_indices,mz))
                        # aggregate intensities accordingly to the new indices:
    #intensity = np.log(intensity) # instead of old way of normalization by dividing max intensity(line53-54), decided to use Log
    
    mz,intensity = _rescale_spectrum(mz,intensity)
    
    
    #intensity = np.log(intensity) # instead of old way of normalization by dividing max intensity(line53-54), decided to use Log

    
    # mz,intensity -> dense matrix of fixed m/z-range populated with intensities:
    spectrum_dense = _to_sparse(mz,intensity)
                        # normalize by maximum intensity
        
    #max_int = np.max(intensity)
    #spectrum_dense = spectrum_dense/max_int
        
    
    #print(spectrum)
    #### PREPROCESSING END #########
    return ID,spectrum_dense

def mgfentry_2_spectrum(mgf_entry):
    ident='title'
    ID = mgf_entry['params'][ident]
    mz = mgf_entry['m/z array']
    intensity =  mgf_entry['intensity array']

    return [ID,mz,intensity]

class create_tf_data(object):
    def __init__(self,datasource='mgf',n=100, MZ_MAX = 2000, SPECTRUM_RESOLUTION = 10, mgf_filename = None, score_filename = None,tfrecord_file='tmp.tfrecord'):
        self.SPECTRUM_RESOLUTION = SPECTRUM_RESOLUTION
        self.MZ_MAX = MZ_MAX
        
        # Get the number of elements from the meta-file, connected to the tfrecord file
        try:
            self.n = int(pd.read_csv(tfrecord_file+'.meta')['n'].values[0])
        except:
            self.n = n

        self.score_filename = score_filename
        self.mgf_filename = mgf_filename  
        self.tfrecord_file = tfrecord_file

        if self.score_filename is not None:
            self.id_score_dict = self._read_labels(self.score_filename)

        if (self.mgf_filename is not None) and (type(self.mgf_filename) is str):
            self._read_spectra_iterative([self.mgf_filename])

        if (self.mgf_filename is not None) and type(self.mgf_filename) == list:
            self.id_score_dict = dict(zip(self.mgf_filename,range(len(self.mgf_filename))))
            pd.DataFrame.from_dict(self.id_score_dict,orient='index').to_csv('labels.file',header=False)
            #self._read_spectra_iterative(self.mgf_filename)

            # read multile mgfs:
            features,spectra = self._read_multiple_mgfs(self.mgf_filename)
            # map filenames to categorical/scalar features, like 0,1,2,... 
            features = [self.id_score_dict[i] for i in features] 
            # write to tfrecordfile
            self._write_tfrecordfile(features, spectra)
    
    def _print_progress(self,progress):
        import sys
        sys.stdout.write("Progress: %d%%   \r" % (progress*100) )
        sys.stdout.flush()
        return None        
        
    def _wrap_int64(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _wrap_float(self,value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _wrap_bytes(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _parse(self,serialized):
        # Define a dict with the data-names and types we expect to
        # find in the TFRecords file.
        # It is a bit awkward that this needs to be specified again,
        # because it could have been written in the header of the
        # TFRecords file instead.
        features = \
            {
                'spectrum': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.float32)
            }

        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)

        # Get the image as raw bytes.
        element_raw = parsed_example['spectrum']

        # Decode the raw bytes so it becomes a tensor with type.
        element = tf.decode_raw(element_raw, tf.float64)
        #element = tf.decode_raw(element_raw, tf.uint8)
        
        # The type is now uint8 but we need it to be float.
        #element = tf.cast(element, tf.int64)

        # Get the label associated with the image.
        label = parsed_example['label']

        return element, label

        
    def _read_spectra_iterative(self,filenames,ident='title'):
        
        out_path = self.tfrecord_file
        print('Processing spectra, may take a minute...')
        # read, process and write spectra one-by-one.
        with tf.python_io.TFRecordWriter(out_path) as writer:
            for filename in filenames:
                with mgf.read(filename,convert_arrays=0) as reader:
                    for i,spectrum in enumerate(reader):
                        if i>self.n:
                            break
                        self._print_progress(i/self.n)

                        ID = spectrum['params'][ident]
                        mz = spectrum['m/z array']
                        intensity =  spectrum['intensity array']

                        #### PREPROCESSING BEGIN #######
                        # round indices according to spectrum resolution, self.SPECTRUM_RESOLUTION:
                        mz = list(map(self._parse_indices,mz))
                        # aggregate intensities accordingly to the new indices:
                        mz,intensity = self._rescale_spectrum(mz,intensity)
                        
                        # mz,intensity -> dense matrix of fixed m/z-range populated with intensities:
                        spectrum_sparse = self._to_sparse(mz,intensity)
                        # normalize by maximum intensity
                        max_int = np.max(intensity)
                        spectrum_sparse = spectrum_sparse/max_int
                        spectrum_as_bytes = spectrum_sparse.tobytes()
                        #print(spectrum)
                        #### PREPROCESSING END #########
                        
                        # get label/score:
                        if len(filenames) == 1:
                            try:
                                label = self.id_score_dict[ID]
                            except: 
                                label = -1
                        else:
                            label = self.id_score_dict[filename]
                        #except: 
                        #    label = -1
                        # Create a dict with the data we want to save in the
                        # TFRecords file. 
                        data = \
                            {
                                'spectrum': self._wrap_bytes(spectrum_as_bytes),
                                'label': self._wrap_float(label)
                            }

                        # Wrap the data as TensorFlow Features.
                        feature = tf.train.Features(feature=data)

                        # Wrap again as a TensorFlow Example.
                        example = tf.train.Example(features=feature)

                        # Serialize the data.
                        serialized = example.SerializeToString()

                        # Write the serialized data to the TFRecords file.
                        writer.write(serialized)             
                
        return None

    def _read_multiple_mgfs(self,filenames,ident='title'):
        #filenames = ['lib_mgf.test','lib_mgf.valid']
        #filename_feature_dict = dict(zip(files,range(len(filenames))))
        print('Reading and indexing %s mgf files...'%(len(filenames)))

        def index_mgfs(filename):
            return [filename,mgf.read(filename, use_index=True)]

        def get_ids_from_mgf(indexed_mgf):
            return indexed_mgf.index.keys()

        # filename assigned to its indexed-mgf object
        feature_indexed_mgfs_dict = dict(list(map(index_mgfs,filenames)))

        # Merge all indexed mgfs to a joint-indexed-mgf containing all spectra
        # Apply a preprocessing function to each spectra, using pyteomics-map on an indexed_mgf
        # Create a tiled-feature vector, this makes sure that features are in the respective order.
        modified_indexed_mgfs = []
        features = []
        for feature,x_mgf in feature_indexed_mgfs_dict.items():
            modified_indexed_mgfs.extend(x_mgf.map(mgfentry_2_spectrum))
            features.extend(np.repeat(feature,len(x_mgf)))

        # Zip feature and vector of all spectra, to keep the assigned feature when shuffling.
        result = list(zip(features,modified_indexed_mgfs))
        # Shuffle: 
        np.random.shuffle(result)

        # Unzip:
        features,spectra = list(zip(*result))

        return features,spectra

    def _write_tfrecordfile(self,features,spectra):
        out_path = self.tfrecord_file
        # write a meta info file for the tfrecord-file, to keep the number of elements in the tfrecord-file   
        pd.DataFrame({'filename':[str(out_path)],'n':[str(len(features))]}).to_csv(out_path+'.meta',index=False)

        print('Writing tfrecordfile to %s ...'%(out_path))
        n = len(features)
        with tf.python_io.TFRecordWriter(out_path) as writer:
            for i,[feature,spectrum] in enumerate(zip(features,spectra)):
                self._print_progress(i/n)
                label = feature
                ID,mz,intensity = spectrum
                _, spectrum_dense = preprocess_spectrum(ID,mz,intensity)



                spectrum_as_bytes = spectrum_dense.tobytes()
                data = {
                        'spectrum': self._wrap_bytes(spectrum_as_bytes),
                        'label': self._wrap_float(label)
                        }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.
                writer.write(serialized)
        return None

    def dump_to_tfrecordfile(self,tfrecord_file,elements,labels):
        out_path = tfrecord_file
        print('Writing tfrecordfile to %s ...'%(out_path))
        n = len(labels)
        with tf.python_io.TFRecordWriter(out_path) as writer:
            for i,[label,element] in enumerate(zip(labels,elements)):
                self._print_progress(i/n)
                
                element_as_bytes = element.tobytes()
                data = {
                        'spectrum': self._wrap_bytes(element_as_bytes),
                        'label': self._wrap_float(label)
                        }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.
                writer.write(serialized)
        return None                     

    
    def get_batch(self,train=True, batch_size=32, buffer_size=2048):
        # Args:
        # filenames:   Filenames for the TFRecords files.
        # train:       Boolean whether training (True) or testing (False).
        # batch_size:  Return batches of this size.
        # buffer_size: Read buffers of this size. The random shuffling
        #              is done on the buffer, so it must be big enough.

        # Create a TensorFlow Dataset-object which has functionality
        # for reading and shuffling data from TFRecords files.
        dataset = tf.data.TFRecordDataset(filenames=self.tfrecord_file)

        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for the image and labels.
        dataset = dataset.map(self._parse)

        if train:
            # If training then read a buffer of the given size and
            # randomly shuffle it.
            dataset = dataset.shuffle(buffer_size=buffer_size)

            # Allow infinite reading of the data.
            num_repeat = None
        else:
            # If testing then don't shuffle the data.

            # Only go through the data once.
            num_repeat = 1

        # Repeat the dataset the given number of times.
        dataset = dataset.repeat(num_repeat)

        # Get a batch of data with the given size.
        dataset = dataset.batch(batch_size)

        # Create an iterator for the dataset and the above modifications.
        iterator = dataset.make_one_shot_iterator()

        # Get the next batch of X and labels.
        Xs_batch, labels_batch = iterator.get_next()

        return Xs_batch, labels_batch

    def get_all(self):
        # Args:
        # filenames:   Filenames for the TFRecords files.
        # train:       Boolean whether training (True) or testing (False).
        # batch_size:  Return batches of this size.
        # buffer_size: Read buffers of this size. The random shuffling
        #              is done on the buffer, so it must be big enough.

        # Create a TensorFlow Dataset-object which has functionality
        # for reading and shuffling data from TFRecords files.
        dataset = tf.data.TFRecordDataset(filenames=self.tfrecord_file)

        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for the image and labels.
        dataset = dataset.map(self._parse)
        
        # Get a batch of data with the given size.
        some_big_number = np.iinfo(np.int32).max
        dataset = dataset.batch(some_big_number)

        # Create an iterator for the dataset and the above modifications.
        iterator = dataset.make_one_shot_iterator()

        # Get the next batch of X and labels.
        elements,labels = iterator.get_next()

        return elements,labels

    def _read_labels(self,filename):        
        df = pd.read_csv(filename,delimiter='\t')
        df['scan'] = pd.DataFrame(df['scan'],dtype='str')
        print('Getting the labels, may take a minute...')
        df = df.groupby('scan')['predicted_score'].nlargest(1).reset_index(level=0)    
        id_label_dict=dict(zip(df['scan'],df['predicted_score']))
        return id_label_dict

    def _parse_indices(self,element):
        resolution = self.SPECTRUM_RESOLUTION
        element = round(element,resolution)
        element = int(element * (10**resolution))
        return [element]

    def _rescale_spectrum(self,indices,values):
        # get unique indices, and positions in the array
        y,idx = np.unique(indices,return_index=True)
        
        # Use the positions of the unique values as the segment ids to sum segments up:
        values = np.add.reduceat(values, idx)
        
        ## Truncate
        mask = np.less(y,self.MZ_MAX * (10**self.SPECTRUM_RESOLUTION))
        indices = y[mask]
        values = values[mask]
        
        #make nested list [1, 2, 3] -> [[1],[2],[3]], as later requiered by SparseTensor:
        #indices = tf.reshape(indices, [tf.size(indices),1])        

        return indices,values
    
    def _to_sparse(self,indices,values):
        from scipy.sparse import csr_matrix
        
        #dim = self.MZ_MAX
        zeros = np.zeros(len(indices),dtype=np.int32)
        
        intensities_array = csr_matrix((values,(indices,zeros)),shape=(self.MZ_MAX * (10**self.SPECTRUM_RESOLUTION),1), dtype=np.float64).toarray().flatten()
        return intensities_array

def main():
    #data = create_tf_data(n=1000,mgf_filename = './example/example.mgf', score_filename = './example/example.deepnovo_denovo')
    
    #multiple mgf files, labels are categories: one cat per mgf-file in the list.
    #list_of_mgf_files = ['./example/example.mgf','./example/peprec_CID_predictions.mgf','./example/peprec_ETD_predictions.mgf','./example/peprec_HCDch2_predictions.mgf','./example/peprec_HCD_predictions.mgf']  
    #data = create_tf_data(n=1000/len(list_of_mgf_files), mgf_filename = list_of_mgf_files)

    #list_of_mgf_files = ['/data/scratch/altenburgt/PXD001374/results/labeled_modified.mgf','/data/scratch/altenburgt/PXD001374/results/labeled_unmodified.mgf']
    list_of_mgf_files = ['./example/example.mgf','./example/peprec_HCD_predictions.mgf']
    data = create_tf_data(mgf_filename = list_of_mgf_files)

    #data = create_tf_data()
    #with tf.Session() as sess:
    #    x,l = sess.run(data.get_all())
    #    print(len(l))

    # Or this, for directly reading tfrecords file
    #data = create_tf_data(mgf_filename = list_of_mgf_files)

if __name__ == '__main__':
    main()
