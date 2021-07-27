using System;
using System.Collections.Generic;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;

using System.Linq;
using System.Collections;
using System.Reflection;
using System.Reflection.Metadata;

namespace ML.Yuk
{
    public class NDArrayEnum : IEnumerator
    {
        public NDArray _rows;

        // Enumerators are positioned before the first element
        // until the first MoveNext() call.
        int position = -1;

        public NDArrayEnum(NDArray list)
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

    public class NDArray : IEnumerable
    {
        private dynamic[] _array;

        public NDArray()
        {
            _array = new dynamic[0];
        }

        // Implementation for the GetEnumerator method.
        IEnumerator IEnumerable.GetEnumerator()
        {
            return (IEnumerator)GetEnumerator();
        }

        public NDArrayEnum GetEnumerator()
        {
            return new NDArrayEnum(this);
        }

        public NDArray(params dynamic[] array)
        {
            _array = array;

            if (array.Length == 1)
            {
                if (array[0] is Array)
                {
                    int size = array[0].Length;
                    dynamic[] a = new dynamic[size];

                    array[0].CopyTo(a, 0);

                    _array = a;
                }
            }
        }

        public void Init(dynamic value, int length)
        {
            for (int i = 0; i < length; i++)
            {
                Add(value);
            }
        }

        public T[] AsType<T>()
        {
            T[] a = _array.Cast<T>().ToArray();

            return a;
        }

        public NDArray Shape()
        {
            NDArray a = Shape(_array);

            return a;
        }

        private NDArray Shape(dynamic array)
        {
            NDArray a = new NDArray();


            if (isArray(array))
            {
                if (array.Length > 0)
                {
                    a.Add(array.Length);

                    NDArray n = Shape(array[0]);
                    if (n.Length != 0)
                    {
                        a = a.Concat(n);
                    }
                }
            }

            return a;
        }

        public bool Equals(NDArray obj)
        {
            bool valid = true;

            dynamic[] a = ToArray();

            valid = SequenceEqual(a, obj.ToArray());

            return valid;
        }

        private bool SequenceEqual(dynamic[] obj1, dynamic[] obj2)
        {
            bool isEqual = true;

            int c1 = obj1.Length;
            int c2 = obj2.Length;

            if (c1 != c2)
            {
                isEqual = false;
                return isEqual;
            }

            for (int i = 0; i < c1; i++)
            {
                dynamic a = obj1[i];
                dynamic b = obj2[i];

                if (a == null || b == null)
                {
                    if (!(a == null && b == null)) 
                    {
                        isEqual = false;
                    }
                }
                else
                {
                    if (!a.Equals(b))
                    {
                        isEqual = false;
                    }
                }
            }

            return isEqual;
        }

        private dynamic[] ToArray()
        {
            return _array;
        }

        public dynamic this[params int[] index]
        {
            get => Get(index);
            set => Set(index, value);
        }

        private void Set(int[] index, dynamic value)
        {
            int i = 0;
            dynamic[] a = _array;

            Set(a, index, i, value);
        }

        private void Set(dynamic array, int[] index, int i, dynamic value)
        {
            if (i < index.Length - 1)
            {
                int j = index[i];

                if (isArray(array))
                {
                    if (array.Length > 0)
                    {
                        dynamic s = array[j];
                        Set(s, index, ++i, value);
                    }
                }
            }
            else
            {
                int j = index[i];
                array[j] = value;
            }
        }

        private dynamic Get(int[] index)
        {
            int i = 0;
            dynamic[] a = _array;

            return Get(a, index, i);
        }

        private dynamic Get(dynamic array, int[] index, int i)
        {
            if (i < index.Length)
            {
                if (isArray(array))
                {
                    if (array.Length > 0)
                    {
                        int j = index[i];
                        dynamic s = array[j];
                        return Get(s, index, ++i);
                    }
                }
            }

            return array;
        }

       
        public dynamic this[params Slice[] index]
        {
            get => GetSlice(index);
        }

        public dynamic this[params Range[] index]
        {
            get => GetRange(index);
        }
        
        private dynamic GetRange(params Range[] index)
        {
            Slice[] a = new Slice[index.Length];

            for (int i = 0; i < index.Length; i++)
            {
                a[i] = new Slice(index[i].Start, index[i].End);
            }

            return GetSlice(a);
        }

        private dynamic GetSlice(params Slice[] index)
        {
            NDArray t = new NDArray();

            for (int i = 0; i < index.Length; i++)
            {
                NDArray a = new NDArray();

                int start = GetIndex(index[i].Start);
                int end = GetIndex(index[i].End);

                if (i < _array.Length)
                {
                    dynamic at = _array;
                   
                    dynamic tmp = _array[i];

                    if (isArray(tmp))
                    {
                        if (tmp.Length > 0)
                        {
                            at = tmp;
                        }
                    }

                    for (int j = 0; j < at.Length; j++)
                    {
                        if (j >= start && j < end)
                        {
                            a.Add(at[j]);
                        }
                    }

                    if (isArray(tmp))
                    {
                        t.Add(a);
                    }
                    else
                    {
                        t = t.Concat(a);
                    }
                }
            }

            return t;
        }

        public NDArray Concat(NDArray array)
        {
            dynamic[] a = Concat(array.ToArray());

            NDArray d = new NDArray(a);

            return d;
        }

        private dynamic[] Concat(dynamic[] array)
        {
            int size = _array.Length + array.Length;
            dynamic[] a = new dynamic[size];

            _array.CopyTo(a, 0);
            array.CopyTo(a, _array.Length);

            return a;
        }

        public void Remove(int index)
        {
            int size = _array.Length - 1;
            dynamic[] a = new dynamic[size];


            Array.Copy(_array, 0, a, 0, index);
            Array.Copy(_array, index + 1 , a, index , size - index);

            _array = a;
        }

        private int GetIndex(Index index)
        {
            int i = index.Value;

            if (index.IsFromEnd)
            {
                i = this.Length - index.Value;
            }

            return i;
        }

        public int Length
        {
            get { return _array.Length; }
        }

        public void Add(dynamic item)
        {
            int size = _array.Length + 1;
            dynamic[] a = new dynamic[size];

            _array.CopyTo(a, 0);

            a[size - 1] = item;

            _array = a;
        }

        // To Do: Deep Copy
        public NDArray Copy()
        {
            int size = _array.Length;
            dynamic[] a = new dynamic[size];

            _array.CopyTo(a, 0);

            NDArray t = new NDArray(a);

            return t;
        }

        private bool isString(dynamic array)
        {
            if (array.GetType() == typeof(String))
            {
                return true;
            }

            return false;
        }

        private bool isArray(dynamic array)
        {
            if (array == null)
            {
                return false;
            }

            Type valueType = array.GetType();

            if (valueType.IsArray || valueType == typeof(NDArray))
            {
                return true;
            }

            return false;
        }


        public static NDArray Unique(NDArray array)
        {
            dynamic[] items = array._array;
            IEnumerable<dynamic> uniqueItems = items.Distinct<dynamic>();

            NDArray newItems = new NDArray(uniqueItems.ToArray());

            return newItems;
        }

        public static double Std(NDArray array)
        {
            double mean = NDArray.Mean(array);

            dynamic sum = 0;

            for (int i = 0; i < array.Length; i++)
            {
                double? t = array[i];

                if (t != null)
                {
                    double? square = Math.Pow((t.Value - mean), 2);

                    sum = sum + square;
                }
            }

            double mean_square = sum / array.Length;

            double sqrt = Math.Sqrt(mean_square);

            return sqrt;
        }

        public static double Mean(NDArray array)
        {
            double sum = 0;

            for (int i = 0; i < array.Length; i++)
            {
                double? t = array[i];

                if (t != null)
                {
                    sum = sum + t.Value;
                }
            }

            double mean = sum / array.Length;

            return mean;
        }

        public static double Sum(NDArray array)
        {
            double sum = 0;

            for (int i = 0; i < array.Length; i++)
            {
                double? t = array[i];

                if (t != null)
                {
                    sum = sum + t.Value;
                }
            }

            return sum;
        }

        public static double Max(NDArray array)
        {
            double? j = Double.NegativeInfinity;

            for (int i = 0; i < array.Length; i++)
            {
                double? t = array[i];

                if ( t >= j)
                {
                    j = t;
                }
            }

            return j.Value;
        }

        public static double Min(NDArray array)
        {
            double? j = Double.PositiveInfinity;

            for (int i = 0; i < array.Length; i++)
            {
                double? t = array[i];

                if (t <= j)
                {
                    j = t;
                }
            }

            return j.Value;
        }

        public NDArray Maxlextrema(int window)
        {
            NDArray t = new NDArray();


            return t;
        }

        public NDArray Minlextrema(int window)
        {
            NDArray t = new NDArray();


            return t;
        }

        public Window Rolling(int window)
        {
            NDArray array = Copy();
            int max = array.Length;
            NDArray narray = new NDArray();

            for (int i = 0; i < max; i++)
            {
                if (i >= window - 1)
                {
                    Slice s = new Slice(i - window + 1, i + 1);
                    NDArray t = array[s];

                    narray.Add(t);
                }
                else
                {
                    narray.Add(null);
                }
            }

            Window win = new Window(narray);

            return win;
        }

        /*
        public void Zeros()
        {

        }

        public int Ndim()
        {

        }

        public int Size()
        {

        }

        public void Arange()
        {

        }

        public void Reshape()
        {

        }
        */
    }
}
