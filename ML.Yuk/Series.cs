using System;
using System.Collections.Generic;
using System.Text;

namespace ML.Yuk
{
    public class Series
    {
        NDArray _array;
        NDArray _index;
        string _colName;
        string _indexName;

        public Series()
        {
            _array = new NDArray();
            _index = new NDArray();

            _colName = "";
            _indexName = "";
        }

        public Series(NDArray array, string colName = null, NDArray index = null, string indexName = null)
        {
            _array = array;

            if (index == null)
            {
                index = CreateIndex();
            }

            _index = index;

            _colName = colName;
            _indexName = indexName;
        }

        public T[] AsType<T>()
        {
            T[] a = _array.AsType<T>();

            return a;
        }

        public NDArray GetValue()
        {
            return _array.Copy();
        }

        public NDArray GetIndex()
        {
            return _index.Copy();
        }

        public string ColName
        {
            get => _colName;
            set => _colName = value;
        }

        public string IndexName
        {
            get => _indexName;
            set => _indexName = value;
        }

        public void SetIndex(NDArray index)
        {
            _index = index;
        }

        private NDArray CreateIndex()
        {
            NDArray nd = new NDArray();

            for (int i = 0; i < _array.Length; i++)
            {
                nd.Add(i);
            }

            return nd;
        }

        public dynamic this[int index]
        {
            get => Get(index);
            set => Set(index, value);
        }

        public dynamic this[string index]
        {
            get => Get(index);
            set => Set(index, value);
        }

        public dynamic this[Slice index]
        {
            get => Get(index);
        }

        private dynamic Get(int index)
        {
            return _array[index];
        }

        private dynamic Get(Slice index)
        {
            return _array[index];
        }

        private dynamic Get(string index)
        {
            int i = FindIndex(index);

            if (i != -1)
            {
                return _array[i];
            }

            return null;
        }

        private void Set(int index, dynamic value)
        {
            _array[index] = value;
        }

        private void Set(String index, dynamic value)
        {
            int i = FindIndex(index);

            if (i != -1)
            {
                _array[i] = value;
            }
        }

        public int FindIndex(dynamic index)
        {
            for (int i = 0; i < _array.Length; i++)
            {
                dynamic t = _index[i];
                if (index.Equals(t))
                {
                    return i;
                }
            }

            return -1;
        }

        public bool Equals(Series obj)
        {
            return _array.Equals(obj._array) && _index.Equals(obj._index) && _colName.Equals(obj._colName) && _indexName.Equals(obj._indexName);
        }

        public int Length
        {
            get { return _index.Length; }
        }

        public void Add(dynamic item, dynamic index=null)
        {
            if (index == null)
            {
                index = this.Length;
            }

            _index.Add(index);
            _array.Add(item);
        }

        public dynamic Max()
        {
            dynamic max = _array[0];

            for (int i = 0; i < _array.Length; i++)
            {
                dynamic t = _array[i];
                if (max < t)
                {
                    max = t;
                }
            }

            return max;
        }
    }
}
