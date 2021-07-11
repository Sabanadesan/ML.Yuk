using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Net;
using System.Collections;

namespace ML.Yuk
{
    public class DataFrameEnum : IEnumerator
    {
        public DataFrame _rows;

        // Enumerators are positioned before the first element
        // until the first MoveNext() call.
        int position = -1;

        public DataFrameEnum(DataFrame list)
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

        public DataFrame Current
        {
            get
            {
                try
                {
                    return _rows.GetRowVal(position);
                }
                catch (IndexOutOfRangeException)
                {
                    throw new InvalidOperationException();
                }
            }
        }
    }

    public class DataFrame : IEnumerable
    {
        private NDArray _data;
        private NDArray _indexes;
        private NDArray _columns;
        private string _indexColumn;

        public DataFrame(string indexColumn = null)
        {
            InitDataFrame();
        }

        // Implementation for the GetEnumerator method.
        IEnumerator IEnumerable.GetEnumerator()
        {
            return (IEnumerator)GetEnumerator();
        }

        public DataFrameEnum GetEnumerator()
        {
            return new DataFrameEnum(this);
        }

        public void InitDataFrame(string indexColumn = null)
        {
            _data = new NDArray();
            _indexes = new NDArray();
            _columns = new NDArray();
            _indexColumn = indexColumn;
        }

        public DataFrame(NDArray data, NDArray index, NDArray columns, string indexColumn)
        {
            _data = data;
            _indexes = index;
            _columns = columns;
            _indexColumn = indexColumn;
        }

        public string IndexColumn
        {
            get { return _indexColumn; }
            set { _indexColumn = value; }
        }

        public NDArray GetIndex()
        {
            return _indexes.Copy();
        }

        public DataFrame(params Series[] pars)
        {
            InitDataFrame();

            for (int i = 0; i < pars.Length; i++)
            {
                Series s = pars[i];

                String k = s.ColName;

                _columns.Add(k);
                _data.Add(s);

                _indexes = NDArray.Unique(_indexes.Concat(s.GetIndex()));

                this.IndexColumn = s.IndexName;
            }
        }

        public void SetIndex(string column)
        {
            int i = FindIndexCol(column);

            Series col = _data[i];

            _data.Remove(i);
            _columns.Remove(i);

            NDArray new_index = col.GetValue();

            for (int j = 0; j < _data.Length; j++)
            {
                Series scol = _data[j];

                NDArray col_index = scol.GetIndex();

                NDArray index = Map(_indexes, col_index, new_index);

                scol.SetIndex(index);
                scol.IndexName = column;
            }

            _indexes = new_index;
        }

        private NDArray Map(NDArray old_index, NDArray col_index, NDArray new_index)
        {
            NDArray t = new NDArray();

            for (int i = 0; i < col_index.Length; i++)
            {
                int index = FindInArray(old_index, col_index[i]);

                if (index != -1)
                {
                    t.Add(new_index[index]);
                }
            }

            return t;
        }

        private int FindInArray(NDArray array, dynamic item)
        {
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] == item)
                {
                    return i;
                }
            }

            return -1;
        }

        public void AddColumn(Series s)
        {
            String k = s.ColName;

            _columns.Add(k);
            _data.Add(s);

            //To Do: Get all indexes and unique
            _indexes = NDArray.Unique(_indexes.Concat(s.GetIndex()));

            this.IndexColumn = s.IndexName;
        }

        public DataFrame GetRowVal(int row)
        {
            NDArray array = new NDArray();

            for (int i = 0; i < _data.Length; i++)
            {
                Series scol = _data[i];
                dynamic a = scol[row];
                array.Add(a);
            }

            DataFrame df = new DataFrame();
            df.Add(array, new NDArray(_indexes[row]), _columns.Copy());

            return df;
        }

        public DataFrame GetRowIndex(dynamic index)
        {
            int row = FindIndexRow(index);

            NDArray array = new NDArray();

            for (int i = 0; i < _data.Length; i++)
            {
                Series scol = _data[i];
                dynamic a = scol[row];
                array.Add(a);
            }

            DataFrame df = new DataFrame();
            df.Add(array, new NDArray(_indexes[row]), _columns.Copy());

            return df;
        }

        private DataFrame GetCol(string column)
        {
            int i = FindIndexCol(column);

            return GetColByIndex(i);
        }

        private DataFrame GetColByIndex(int column)
        {
            Series scol = _data[column];

            DataFrame df = new DataFrame();
            df.AddColumn(scol);

            return df;
        }

        private Series GetColBySeries(string column)
        {
            int i = FindIndexCol(column);

            return GetColBySeriesByIndex(i);
        }

        private Series GetColBySeriesByIndex(int column)
        {
            Series scol = _data[column];

            return scol;
        }

        private void SetCol(string column, NDArray value)
        {
            int i = FindIndexCol(column);

            SetColByIndex(i, value, column);
        }

        private void SetColByIndex(int column, NDArray value, string col = null)
        {
            if (column == -1)
            {
                if (col == null)
                {
                    throw new System.IndexOutOfRangeException();
                }

                Series s = new Series();
                s.ColName = col;

                _data.Add(s);

                _columns.Add(col);

                column = _columns.Length - 1;
            }

            Series t = new Series(value);
            t.ColName = col;

            _data[column] = t;

            _indexes = NDArray.Unique(_indexes.Concat(t.GetIndex()));
        }

        private void SetCol(string column, DataFrame value)
        {
            int i = FindIndexCol(column);

            SetColByIndex(i, value, column);
        }

        private void SetCol(string column, Series value)
        {
            int i = FindIndexCol(column);

            SetColByIndex(i, value, column);
        }

        private void SetColByIndex(int column, DataFrame value, string col = null)
        {
            if (column == -1)
            {
                if (col == null)
                {
                    throw new System.IndexOutOfRangeException();
                }

                Series s = new Series();
                s.ColName = col;

                _data.Add(s);

                _columns.Add(col);

                column = _columns.Length - 1;
            }

            NDArray t = value.GetValue();
            _data[column] = new Series(t[0], col);

            _indexes = NDArray.Unique(_indexes.Concat(value.GetIndex()));
        }

        private void SetColByIndex(int column, Series value, string col = null)
        {
            if (column == -1)
            {
                if (col == null)
                {
                    throw new System.IndexOutOfRangeException();
                }

                Series s = new Series();
                s.ColName = col;

                _data.Add(s);

                _columns.Add(col);

                column = _columns.Length - 1;
            }

            _data[column] = value;

            _indexes = NDArray.Unique(_indexes.Concat(value.GetIndex()));
        }

        public dynamic this[string col]
        {
            get => GetCol(col);
            set
            {
                if (value == null)
                {
                    NDArray vNDArray = new NDArray();
                    vNDArray.Init(value, this.Length);
                    SetCol(col, vNDArray);
                }
                else
                {
                    Type valueType = value.GetType();

                    if (valueType == typeof(NDArray))
                    {
                        NDArray vNDArray = value;
                        SetCol(col, vNDArray);
                    }
                    else if (valueType == typeof(Series))
                    {
                        Series vSeries = value;
                        SetCol(col, vSeries);
                    }
                    else if (valueType == typeof(DataFrame))
                    {
                        DataFrame vDataFrame = value;
                        SetCol(col, vDataFrame);
                    }
                    else
                    {
                        NDArray vNDArray = new NDArray();
                        vNDArray.Init(value, this.Length);
                        SetCol(col, vNDArray);
                    }
                }

            }
        }

        public dynamic this[int col]
        {
            get => GetColByIndex(col);
            set
            {
                if (value == null)
                {
                    NDArray vNDArray = new NDArray();
                    vNDArray.Init(value, this.Length);
                    SetColByIndex(col, vNDArray);
                }
                else
                {
                    Type valueType = value.GetType();

                    if (valueType == typeof(NDArray))
                    {
                        NDArray vNDArray = value;
                        SetColByIndex(col, vNDArray);
                    }
                    else if (valueType == typeof(Series))
                    {
                        Series vSeries = value;
                        SetColByIndex(col, vSeries);
                    }
                    else if (valueType == typeof(DataFrame))
                    {
                        DataFrame vDataFrame = value;
                        SetColByIndex(col, vDataFrame);
                    }
                }
            }
        }

        public dynamic this[Slice row, Slice col]
        {
            get => Get(row, col);
        }

        public dynamic this[string row, string col]
        {
            get => Get(FindIndexRow(row), FindIndexCol(col));
            set => Set(FindIndexRow(row), FindIndexCol(col), value);
        }

        public dynamic this[int row, int col]
        {
            get => Get(row, col);
            set => Set(row, col, value);
        }

        public dynamic this[String row, int col]
        {
            get => Get(FindIndexRow(row), col);
            set => Set(FindIndexRow(row), col, value);
        }

        public dynamic this[int row, String col]
        {
            get => Get(row, FindIndexCol(col));
            set => Set(row, FindIndexCol(col), value);
        }

        public dynamic this[Slice row, String col]
        {
            get => Get(row, FindIndexCol(col));
        }

        public dynamic this[String row, Slice col]
        {
            get => Get(FindIndexRow(row), col);
        }

        public dynamic this[Slice row, int col]
        {
            get => Get(row, col);
        }

        public dynamic this[int row, Slice col]
        {
            get => Get(row, col);
        }

        private dynamic Get(int row, int col)
        {
            Series scol = _data[col];

            if (row < 0)
            {
                row = scol.Length + row;
            }

            dynamic t = scol[row];

            return t;
        }

        private DataFrame Get(Slice row, int col)
        {
            Series scol = _data[col];
            NDArray t = scol[row];

            DataFrame df = new DataFrame();

            df.AddColumn(new Series(t, _columns[col], _indexes[row]));

            return df;
        }

        private DataFrame Get(int row, Slice col)
        {
            int startCol = GetIndex(col.Start);
            int endCol = GetIndex(col.End);

            DataFrame df = new DataFrame();

            for (int i = 0; i < _data.Length; i++)
            {
                if (i >= startCol && i < endCol)
                {
                    Series scol = _data[i];
                    dynamic a = scol[row];

                    df.AddColumn(new Series(new NDArray(a), _columns[i], new NDArray(_indexes[row])));
                }
            }

            return df;
        }

        private void Set(int row, int col, dynamic value)
        {
            Series scol = _data[col];
            scol[row] = value;
        }

        private DataFrame Get(Slice row, Slice col)
        {
            int startRow = GetIndex(row.Start);
            int endRow = GetIndex(row.End);

            int startCol = GetIndex(col.Start);
            int endCol = GetIndex(col.End);

            DataFrame df = new DataFrame();

            for (int i = 0; i < _data.Length; i++)
            {
                if (i >= startCol && i < endCol)
                {
                    Series scol = _data[i];
                    NDArray a = scol[row];

                    df.AddColumn(new Series(a, _columns[i], _indexes[row]));
                }
            }

            return df;
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
            get { return GetLength(); }
        }

        public NDArray Columns
        {
            get { return _columns.Copy(); }
        }

        private int GetLength()
        {
            NDArray array = _data;

            int max = 0;

            for (int i = 0; i < array.Length; i++)
            {
                Series s = array[i];
                int j = s.Length;

                if (j > max)
                {
                    max = j;
                }
            }

            return max;
        }

        private int FindIndexCol(dynamic index)
        {
            return FindIndex(_columns, index);
        }

        private int FindIndexRow(dynamic index)
        {
            return FindIndex(_indexes, index);
        }

        private int FindIndex(NDArray array, dynamic index)
        {
            for (int i = 0; i < array.Length; i++)
            {
                dynamic t = array[i];
                if (t.Equals(index))
                {
                    return i;
                }
            }

            return -1;
        }

        public bool Equals(DataFrame obj)
        {
            return _data.Equals(obj._data) && _columns.Equals(obj._columns) && _indexes.Equals(obj._indexes);
        }

        public void Add(NDArray data, NDArray index = null, NDArray columns = null, NDArray dataTypes = null, string indexName = null)
        {
            AddFind(data, true, 0, 0, index, columns, dataTypes, indexName);
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

        private void AddFind(dynamic array, bool isRow, int row, int col, NDArray index = null, NDArray columns = null, NDArray dataTypes = null, string indexName = null)
        {
            int max = 0;

            if (isArray(array))
            {
                max = array.Length;
            }

            if (max > 0)
            {
                for (int i = 0; i < max; i++)
                {
                    if (isRow)
                    {
                        AddFind(array[i], false, row, i, index, columns, dataTypes, indexName);
                    }
                    else
                    {
                        AddFind(array[i], true, i, col, index, columns, dataTypes, indexName);
                    }
                }
            }

            if (!isArray(array))
            {
                AddToDataFrame(row, col, array, index, columns, dataTypes, indexName);
            }
        }

        private void AddToDataFrame(int row, int col, dynamic value, NDArray index = null, NDArray columns = null, NDArray dataTypes = null, string indexName = null)
        {
            int i = -1;

            dynamic lbl_col = _columns.Length;
            if (columns != null)
            {
                lbl_col = columns[col];
                i = FindIndexCol(lbl_col);
            }
            else
            {
                i = col;
            }
            
            if (i == -1)
            {
                Series s = new Series();
                s.ColName = lbl_col;
                s.IndexName = indexName;

                _data.Add(s);
                _columns.Add(lbl_col);
                i = FindIndexCol(lbl_col);
            }

            Series t = _data[i];

            int j = -1;
            dynamic lbl_index = _indexes.Length;

            if (index != null)
            {
                lbl_index = index[row];
                j = t.FindIndex(lbl_index);
            }
            else
            {
                j = -1;
            }

            if (j == -1)
            {
                dynamic ind = null;

                if (index != null)
                {
                    ind = lbl_index;
                }

                if (dataTypes != null)
                {
                    Type type = dataTypes[col];
                    AddToSeries(t, value, type, ind);
                }
                else
                {
                   t.Add(value, ind);
                }
                    
                if (col == 0)
                {
                    _indexes.Add(lbl_index);
                }

                j = FindIndexRow(lbl_index);
            }
            else
            {
                if (dataTypes != null)
                {
                    Type type = dataTypes[col];
                    SetToSeries(t, j, value, type);
                }
                else
                {
                    t[j] = value;
                }
            }
        }

        private void AddToSeries(Series array, dynamic value, Type type, dynamic index)
        {
            dynamic new_value = SetType(value, type);
            array.Add(new_value, index);
        }

        private void SetToSeries(Series array, int index, dynamic value, Type type)
        {

            dynamic new_value = SetType(value, type);
            array[index] = new_value;
        }

        private static dynamic SetType(dynamic value, Type type)
        {
            dynamic new_value = null;

            if (type.Equals(typeof(int)))
            {
                new_value = Convert.ToInt32(value);
            }
            else if (type.Equals(typeof(bool)))
            {
                new_value = Convert.ToBoolean(value);
            }
            else if (type.Equals(typeof(double)))
            {
                new_value = Convert.ToDouble(value);
            }
            else if (type.Equals(typeof(DateTime)))
            {
                new_value = Convert.ToDateTime(value);
            }
            else
            {
                new_value = Convert.ToString(value);
            }

            return new_value;
        }

        //
        // Summary:
        //     Reads a text file as a DataFrame. Follows pandas API.
        //
        // Parameters:
        //   filename:
        //     filename
        //
        //   separator:
        //     column separator
        //
        //   header:
        //     has a header or not
        //
        //   columnNames:
        //     column names (can be empty)
        //
        //   dataTypes:
        //     column types (can be empty)
        //
        //   numRows:
        //     number of rows to read
        //
        //   guessRows:
        //     number of rows used to guess types
        //
        //   addIndexColumn:
        //     add one column with the row index
        //
        //   encoding:
        //     The character encoding. Defaults to UTF8 if not specified
        //
        // Returns:
        //     DataFrame
        public static DataFrame LoadCsv(string filename, char separator = ',', bool header = true, NDArray columnNames = null, NDArray dataTypes = null, bool addIndexColumn = false)
        {
            FileStream fs = File.OpenRead(filename);

            DataFrame df = LoadCsv(fs, separator, header, columnNames, dataTypes, addIndexColumn);

            fs.Close();

            return df;
        }

        //
        // Summary:
        //     Reads a seekable stream of CSV data into a DataFrame. Follows pandas API.
        //
        // Parameters:
        //   csvStream:
        //     stream of CSV data to be read in
        //
        //   separator:
        //     column separator
        //
        //   header:
        //     has a header or not
        //
        //   columnNames:
        //     column names (can be empty)
        //
        //   dataTypes:
        //     column types (can be empty)
        //
        //   numberOfRowsToRead:
        //     number of rows to read not including the header(if present)
        //
        //   guessRows:
        //     number of rows used to guess types
        //
        //   addIndexColumn:
        //     add one column with the row index
        //
        //   encoding:
        //     The character encoding. Defaults to UTF8 if not specified
        //
        // Returns:
        //     DataFrame
        public static DataFrame LoadCsv(Stream csvStream, char separator = ',', bool header = true, NDArray columnNames = null, NDArray dataTypes = null, bool addIndexColumn = false)
        {
            String s = Read(csvStream);

            string[] lines = s.Split('\n');

            NDArray array;
            NDArray cols = columnNames;

            NDArray indexVal = null;
            Type type = null;

            DataFrame df;

            if (addIndexColumn)
            {
                string indexColumn = null;

                if (columnNames != null)
                {
                    indexColumn = columnNames[0];
                }

                type = dataTypes[0];

                dataTypes = RemoveIndexFromDataType(dataTypes);

                df = new DataFrame(indexColumn);
            }
            else
            {
                df = new DataFrame();
            }

            for (int i = 0; i < lines.Length - 1; i++)
            {
                array = GetLine(lines[i], addIndexColumn);
                
                if (array.Length > 0)
                {
                    if (header == true && i == 0)
                    {
                        cols = array;
                    }
                    else
                    {
                        if (addIndexColumn)
                        {
                            indexVal = GetLineIndex(lines[i], type);
                        }

                        df.Add(array, indexVal, cols, dataTypes);
                    }
                }
            }

            return df;
        }

        public bool WriteCsv(string filePath, char separator = ',', bool header = true, NDArray columnNames = null)
        {

            NDArray index = GetIndex();

            if (columnNames == null)
            {
                columnNames = this.Columns;
            }

            StringBuilder sbRtn = new StringBuilder();

            if (header)
            {
                // If you want headers for your file
                var strheader = string.Format("\"{0}\"{1}", this.IndexColumn, separator);

                foreach (dynamic col in columnNames)
                {
                    var str = string.Format("\"{0}\"{1}", col, separator);
                    strheader = strheader + str;
                }

                strheader = strheader.Remove(strheader.Length - 1);

                sbRtn.AppendLine(strheader);
            }

            int i = 0;
            foreach (DataFrame row in this)
            {
                // If you want headers for your file
                var strdata = string.Format("\"{0}\"{1}", index[i], separator);

                foreach (dynamic col in columnNames)
                {
                    var str = string.Format("\"{0}\"{1}", row[0, col], separator);
                    strdata = strdata + str;
                }

                strdata = strdata.Remove(strdata.Length - 1);

                sbRtn.AppendLine(strdata);
                i++;
            }
            
            File.WriteAllText(filePath, sbRtn.ToString());

            return true;
        }

        private static NDArray RemoveIndexFromDataType(NDArray dataTypes)
        {
            NDArray nd = new NDArray();

            for(int i = 1; i < dataTypes.Length; i++)
            {
                nd.Add(dataTypes[i]);
            }

            return nd;
        }

        private static NDArray GetLine(String line, bool ignoreIndex = false)
        {
            NDArray nd = new NDArray();

            TextFieldParser parser = new TextFieldParser();

            string[] fields;

            line = line.Replace("\r", "");
            line = line.Replace("\0", "");

            fields = parser.ParseFields(line);
            for(int i = 0; i < fields.Length; i++)
            {
                if (ignoreIndex == true && i == 0)
                {
                    // Skip
                }
                else
                {
                    string field = fields[i];
                    nd.Add(field);
                }
            }

            return nd;
        }

        private static NDArray GetLineIndex(String line, Type type)
        {

            if (type == null)
            {
                type = line.GetType();
            }


            NDArray nd = null;

            TextFieldParser parser = new TextFieldParser();

            string[] fields;

            line = line.Replace("\r", "");
            line = line.Replace("\0", "");

            fields = parser.ParseFields(line);

            if (fields.Length > 0)
            {
                string field = fields[0];

                nd = new NDArray();

                dynamic new_field = SetType(field, type);

                nd.Add(new_field);
            }

            return nd;
        }

        private static String Read(Stream stream)
        {
            byte[] bytes = new byte[stream.Length + 10];
            int numBytesToRead = (int)stream.Length;
            int numBytesRead = 0;

            do
            {
                // Read may return anything from 0 to 10.
                int n = stream.Read(bytes, numBytesRead, 10);
                numBytesRead += n;
                numBytesToRead -= n;
            } while (numBytesToRead > 0);

            String converted = Encoding.UTF8.GetString(bytes, 0, bytes.Length);

            return converted;
        }

        public DataFrame PctChange()
        {
            return ApplyPreviousFunc(this, CalcPctChange);
        }

        public DataFrame CumProd()
        {
            return ApplyPreviousAllFunc(this, CumProduct, true);
        }

        private static double? Sum(double? a, double? b)
        {
            if (a == null || b == null)
            {
                return null;
            }

            return a + b;
        }

        private static double? Minus(double? a, double? b)
        {
            if (a == null || b == null)
            {
                return null;
            }

            return a - b;
        }

        private static double? Product(double? a, double? b)
        {
            if (a == null || b == null)
            {
                return null;
            }

            return a * b;
        }

        private static double? CumProduct(DataFrame a, dynamic column, dynamic index)
        {
            double? total = a[0, column];

            if (total == null)
            {
                total = 1;
            }

            for (int i=1; i <= index; i++)
            {
                double? t = a[i, column];
                if (t == null)
                {
                   t = 1;
                }

                total = total * t;
            }

            if (a[index, column] == null)
            {
                return null;
            }

            return total;
        }

        private static double? Div(double? a, double? b)
        {
            if (a == null || b == null)
            {
                return null;
            }

            return a / b;
        }

        private static double? CalcPctChange(double? a, double? b)
        {
            if (a == null || b == null)
            {
                return null;
            }

            return ((b - a) / a);
        }

        private static DataFrame ApplyFunc(DataFrame a, DataFrame b, Func<double?, double?, double?> c)
        {
            DataFrame z = new DataFrame();

            for (int i = 0; i < a.Columns.Length; i++)
            {
                Series t1 = new Series();

                Series t = a.GetColBySeriesByIndex(i);

                for (int j = 0; j < t.Length; j++)
                {
                    t1.Add(c(a[j, i], b[j, i])); 
                }

                t1.ColName = t.ColName;
                t1.IndexName = t.IndexName;

                z.AddColumn(t1);
            }

            return z;
        }

        private static DataFrame ApplyFuncDouble(DataFrame a, double? b, Func<double?, double?, double?> c)
        {
            DataFrame z = new DataFrame();

            for (int i = 0; i < a.Columns.Length; i++)
            {
                Series t1 = new Series();

                Series t = a.GetColBySeriesByIndex(i);

                for (int j = 0; j < t.Length; j++)
                {
                    t1.Add(c(a[j, i], b));
                }

                t1.ColName = t.ColName;
                t1.IndexName = t.IndexName;

                z.AddColumn(t1);
            }

            return z;
        }

        private static DataFrame ApplyPreviousFunc(DataFrame a, Func<double?, double?, double?> c, bool isDefault = false)
        {
            DataFrame z = new DataFrame();

            for (int i = 0; i < a.Columns.Length; i++)
            {
                Series t1 = new Series();

                Series t = a.GetColBySeriesByIndex(i);

                for (int j = 0; j < t.Length; j++)
                {
                    if (j == 0)
                    {
                        if (isDefault)
                        {
                            t1.Add(a[j, i]);
                        }
                        else
                        {
                            t1.Add(null);
                        }
                    }
                    else
                    {
                        t1.Add(c(a[j-1, i], a[j, i]));
                    }
                }

                t1.ColName = t.ColName;
                t1.IndexName = t.IndexName;

                z.AddColumn(t1);
            }

            return z;
        }

        private static DataFrame ApplyPreviousAllFunc(DataFrame a, Func<DataFrame, dynamic, dynamic, double?> c, bool isDefault = false)
        {
            DataFrame z = new DataFrame();

            for (int i = 0; i < a.Columns.Length; i++)
            {
                Series t1 = new Series();

                Series t = a.GetColBySeriesByIndex(i);

                for (int j = 0; j < t.Length; j++)
                {
                    double? total = c(a, i, j);
                    t1.Add(total);
                }

                t1.ColName = t.ColName;
                t1.IndexName = t.IndexName;

                z.AddColumn(t1);
            }

            return z;
        }

        public static DataFrame operator +(DataFrame a)
        {
            return a;
        }

        public static DataFrame operator -(DataFrame a)
        {
            return a * -1;
        }

        public static DataFrame operator +(DataFrame a, DataFrame b)
        {
            return ApplyFunc(a, b, Sum);
        }

        public static DataFrame operator +(DataFrame a, double b)
        {
            return ApplyFuncDouble(a, b, Sum);
        }

        public static DataFrame operator +(double a, DataFrame b)
        {
            return ApplyFuncDouble(b, a, Sum);
        }

        public static DataFrame operator -(DataFrame a, DataFrame b)
        {
            return a + (-b);
        }

        public static DataFrame operator -(DataFrame a, double b)
        {
            return a + (-b);
        }

        public static DataFrame operator *(DataFrame a, DataFrame b)
        {
            return ApplyFunc(a, b, Product);
        }

        public static DataFrame operator *(DataFrame a, double b)
        {
            return ApplyFuncDouble(a, b, Product);
        }

        public static DataFrame operator *(double a, DataFrame b)
        {
            return ApplyFuncDouble(b, a, Product);
        }

        public static DataFrame operator /(DataFrame a, DataFrame b)
        {
            return ApplyFunc(a, b, Div);
        }

        public static DataFrame operator /(DataFrame a, double b)
        {
            return ApplyFuncDouble(a, b, Div);
        }

        public NDArray GetValue()
        {
            NDArray nd = new NDArray();

            for (int i = 0; i < _data.Length; i++)
            {
                Series col = _data[i];

                NDArray val = col.GetValue();

                nd.Add(val);
            }

            return nd;
        }

        public T[] GetValueCol<T>(string column)
        {
            int i = FindIndexCol(column);

            return GetValueCol<T>(i);
        }

        public T[] GetValueCol<T>(int col)
        {
            Series s = _data[col];

            NDArray val = s.GetValue();

            T[] i = val.AsType<T>();

            return i;
        }
    }
}