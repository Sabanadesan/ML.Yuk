using System;
using System.Collections.Generic;
using System.Text;
using System.Collections;

namespace ML.Yuk
{
    public class SeriesEnum : IEnumerator
    {
        public Series _rows;

        // Enumerators are positioned before the first element
        // until the first MoveNext() call.
        int position = -1;

        public SeriesEnum(Series list)
        {
            _rows = list;
        }

        public bool MoveNext()
        {
            position++;
            return (position < _rows.Length);
        }

        public void Reset()
        {
            position = -1;
        }

        object IEnumerator.Current
        {
            get
            {
                return Current;
            }
        }

        public dynamic Current
        {
            get
            {
                try
                {
                    return _rows[position];
                }
                catch (IndexOutOfRangeException)
                {
                    throw new InvalidOperationException();
                }
            }
        }
    }

    public class Series : IEnumerable
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

        // Implementation for the GetEnumerator method.
        IEnumerator IEnumerable.GetEnumerator()
        {
            return (IEnumerator)GetEnumerator();
        }

        public SeriesEnum GetEnumerator()
        {
            return new SeriesEnum(this);
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
            bool valid = true;

            if (!_array.Equals(obj._array))
            {
                valid = false;
            }

            if (!_index.Equals(obj._index))
            {
                valid = false;
            }

            if (_colName == null && obj._colName == null)
            {

            }
            else
            {
                if (!_colName.Equals(obj._colName))
                {
                    valid = false;
                }
            }

            if (_indexName == null && obj._indexName == null)
            {

            }
            else
            {
                if (!_indexName.Equals(obj._indexName))
                {
                    valid = false;
                }
            }

            return valid;
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

        public Series Append(Series value)
        {
            Series s = this.Copy();
            for(int i = 0; i < value.Length; i++)
            {
                dynamic t = value[i];
                dynamic ix = value._index[i];

                dynamic ix2 = FindIndex(ix);

                if (ix2 == -1) {
                    s.Add(t, ix);
                }
            }

            return s;
        }

        // To Do: Deep Copy
        public Series Copy()
        {
            NDArray array = _array.Copy();
            NDArray index = _index.Copy();

            Series s = new Series(array, ColName, index, IndexName);

            return s;
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
