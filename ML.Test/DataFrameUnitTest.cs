using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

using Xunit;

using System.Linq;
using ML.Yuk;
using System.Xml.Serialization;
using Xunit.Sdk;
using System.Reflection.Metadata.Ecma335;

namespace ML.Test
{
    public class DataFrameUnitTest
    {

        private String GetPath(String symbol, String csvDir)
        {
            // Append symbol name to file extension
            string fileName = symbol + ".csv";

            // Get current directory from build folder
            string currentDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.FullName;

            // Append current directory to csv directory
            string path = Path.Combine(currentDir, csvDir);

            // Append csv directory path to file name
            var filePath = Path.Combine(path, fileName);

            return filePath;
        }

        [Fact]
        public void TestEquals()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            Series col3 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col4 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df1 = new DataFrame(col3, col4);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void TestIndexerGet()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            int i = df[2, 1];

            Assert.True(i.Equals(6), "Arrays are not equal.");
        }

        [Fact]
        public void TestIndexerSet()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            int i = df[2, 1];

            df[2, 1] = 10;

            int j = df[2, 1];

            Assert.True(j.Equals(10), "Arrays are not equal.");
        }

        [Fact]
        public void TestIndexerGetColumn()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            int i = df[1, "Column2"];

            Assert.True(i.Equals(5), "Arrays are not equal.");
        }

        [Fact]
        public void TestIndexerGetSliceCol()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            DataFrame i = df[new Slice(0, 2), "Column2"];

            DataFrame t = new DataFrame(new Series(new NDArray(4, 5), "Column2"));

            Assert.True(i.Equals(t), "Arrays are not equal.");
        }

        [Fact]
        public void TestIndexerGetSliceRow()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            DataFrame i = df[1, new Slice(0, 2)];

            DataFrame t = new DataFrame();
            t.AddColumn(new Series(new NDArray(2), "Column1", new NDArray(1)));
            t.AddColumn(new Series(new NDArray(5), "Column2", new NDArray(1)));

            Assert.True(i.Equals(t), "Arrays are not equal.");
        }

        [Fact]
        public void TestIndexerGetSlice()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            DataFrame i = df[new Slice(1, 3), new Slice(0, 2)];

            DataFrame t = new DataFrame();
            t.AddColumn(new Series(new NDArray(2, 3), "Column1", new NDArray(1, 2)));
            t.AddColumn(new Series(new NDArray(5, 6), "Column2", new NDArray(1, 2)));

            Assert.True(i.Equals(t), "Arrays are not equal.");
        }

        [Fact]
        public void TestAdd()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            NDArray i = new NDArray(new NDArray(7, 8), new NDArray(9, 10));

            df.Add(i);

            Series col3 = new Series(new NDArray(1, 2, 3, 7, 8) , "Column1");
            Series col4 = new Series(new NDArray(4, 5, 6, 9, 10), "Column2");

            DataFrame df1 = new DataFrame(col3, col4);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void TestAddGetIndex()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            NDArray i = new NDArray(new NDArray(7, 8), new NDArray(9, 10));

            df.Add(i);

            Series col3 = new Series(new NDArray(1, 2, 3, 7, 8), "Column1");
            Series col4 = new Series(new NDArray(4, 5, 6, 9, 10), "Column2");

            NDArray index = df.GetIndex();

            Assert.True(index.Equals(new NDArray(0, 1, 2, 3, 4)), "Arrays are not equal.");
        }

        [Fact]
        public void TestAddSingle()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            NDArray i = new NDArray(7, 8);

            df.Add(i);


            Series col3 = new Series(new NDArray(1, 2, 3, 7), "Column1");
            Series col4 = new Series(new NDArray(4, 5, 6, 8), "Column2");

            DataFrame df1 = new DataFrame(col3, col4);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void TestAddIndexColumn()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            NDArray i = new NDArray(new NDArray(7, 8), new NDArray(9, 10));

            NDArray index = new NDArray(1, 2);
            NDArray columns = new NDArray("Column1", "Column2");

            df.Add(i, index, columns);

            Series col3 = new Series(new NDArray(1, 7, 8), "Column1");
            Series col4 = new Series(new NDArray(4, 9, 10), "Column2");

            DataFrame df1 = new DataFrame(col3, col4);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void TestAddNewIndex()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            NDArray i = new NDArray(new NDArray(7, 8), new NDArray(9, 10));

            NDArray index = new NDArray("Hat", "Cat");

            df.Add(i, index, null);

            Series col3 = new Series(new NDArray(1, 2, 3, 7, 8), "Column1", new NDArray(0, 1, 2, "Hat", "Cat"));
            Series col4 = new Series(new NDArray(4, 5, 6, 9, 10), "Column2", new NDArray(0, 1, 2, "Hat", "Cat"));

            DataFrame df1 = new DataFrame(col3, col4);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void TestAddNewIndexGetIndex()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            NDArray i = new NDArray(new NDArray(7, 8), new NDArray(9, 10));

            NDArray index = new NDArray("Hat", "Cat");

            df.Add(i, index, null);

            NDArray t = df.GetIndex();

            Assert.True(t.Equals(new NDArray(0, 1, 2, "Hat", "Cat")), "Arrays are not equal.");
        }

        [Fact]
        public void TestAddIndex()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            NDArray i = new NDArray(new NDArray(7, 8), new NDArray(9, 10));

            NDArray index = new NDArray(1, 2);

            df.Add(i, index, null);

            Series col3 = new Series(new NDArray(1, 7, 8), "Column1");
            Series col4 = new Series(new NDArray(4, 9, 10), "Column2");

            DataFrame df1 = new DataFrame(col3, col4);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void TestAddColumn()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            NDArray i = new NDArray(new NDArray(7, 8), new NDArray(9, 10));

            NDArray columns = new NDArray("Column1", "Column2");

            df.Add(i, null, columns);

            Series col3 = new Series(new NDArray(1, 2, 3, 7, 8), "Column1");
            Series col4 = new Series(new NDArray(4, 5, 6, 9, 10), "Column2");

            DataFrame df1 = new DataFrame(col3, col4);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void ReadCsv()
        {
            String filePath = GetPath("AMZN", "csv");

            NDArray data = new NDArray(typeof(DateTime), typeof(double), typeof(double), typeof(double), typeof(double), typeof(double), typeof(int));

            FileStream fs = File.OpenRead(filePath);

            DataFrame df = DataFrame.LoadCsv(fs, ',', true, null, data);

            fs.Close();

            Series col = new Series(new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14")), "Date", null, "Date");
            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", null, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", null, "Date");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", null, "Date");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", null, "Date");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", null, "Date");
            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", null, "Date");

            DataFrame df1 = new DataFrame(col, col1, col2, col3, col4, col5, col6);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void ReadCsvFile()
        {
            String filePath = GetPath("AMZN", "csv");

            NDArray data = new NDArray(typeof(DateTime), typeof(double), typeof(double), typeof(double), typeof(double), typeof(double), typeof(int));

            DataFrame df = DataFrame.LoadCsv(filePath, ',', true, null, data);

            Series col = new Series(new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14")), "Date", null, "Date");
            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", null, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", null, "Date");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", null, "Date");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", null, "Date");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", null, "Date");
            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", null, "Date");

            DataFrame df1 = new DataFrame(col, col1, col2, col3, col4, col5, col6);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void ReadCsvFileIndex()
        {
            String filePath = GetPath("AMZN", "csv");

            NDArray data = new NDArray(typeof(DateTime), typeof(double), typeof(double), typeof(double), typeof(double), typeof(double), typeof(int));

            DataFrame df = DataFrame.LoadCsv(filePath, ',', true, null, data, true);

            NDArray index = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"));

            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", index, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", index, "Date");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", index, "Date");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", index, "Date");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", index, "Date");
            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", index, "Date");

            DataFrame df1 = new DataFrame(col1, col2, col3, col4, col5, col6);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void ReadCsvFileIndexAndColumns()
        {
            String filePath = GetPath("AMZN", "csv");

            NDArray data = new NDArray(typeof(DateTime), typeof(double), typeof(double), typeof(double), typeof(double), typeof(double), typeof(int));

            NDArray cols = new NDArray("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume");

            DataFrame df = DataFrame.LoadCsv(filePath, ',', true, cols, data, true);

            NDArray index = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"));

            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", index, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", index, "Date");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", index, "Date");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", index, "Date");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", index, "Date");
            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", index, "Date");

            DataFrame df1 = new DataFrame(col1, col2, col3, col4, col5, col6);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void GetRow()
        {
            NDArray index = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"));

            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", index, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", index, "Date");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", index, "Date");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", index, "Date");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", index, "Date");
            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", index, "Date");

            DataFrame df1 = new DataFrame(col1, col2, col3, col4, col5, col6);

            DataFrame t = df1.GetRowIndex(DateTime.Parse("2018-03-13"));

            DataFrame t2 = new DataFrame();
            NDArray cols = new NDArray("Open", "High", "Low", "Close", "Adj Close", "Volume");

            t2.Add(new NDArray(1615.959961, 1617.540039, 1578.010010, 1588.180054, 1588.180054, 6531900), new NDArray(DateTime.Parse("2018-03-13")), cols);

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void GetColumn()
        {
            NDArray index = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"));

            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", index, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", index, "Date");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", index, "Date");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", index, "Date");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", index, "Date");
            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", index, "Date");

            DataFrame df1 = new DataFrame(col1, col2, col3, col4, col5, col6);

            DataFrame t = df1["Low"];

            DataFrame t2 = new DataFrame(new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", index, "Date"));

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void AddColumn()
        {
            NDArray index = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"));

            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", index, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", index, "Date");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", index, "Date");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", index, "Date");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", index, "Date");

            DataFrame df1 = new DataFrame(col1, col2, col3, col4, col5);

            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", index, "Date");

            df1.AddColumn(col6);

            DataFrame t = df1["Volume"];

            DataFrame t2 = new DataFrame(new Series(new NDArray(5174200, 6531900, 4175400), "Volume", index, "Date"));

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void SetIndex()
        {
            Series col = new Series(new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14")), "Date");
            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close");
            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume");

            DataFrame df = new DataFrame(col, col1, col2, col3, col4, col5, col6);

            df.SetIndex("Date");

            NDArray index = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"));

            Series col1a = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", index, "Date");
            Series col2a = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", index, "Date");
            Series col3a = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", index, "Date");
            Series col4a = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", index, "Date");
            Series col5a = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", index, "Date");
            Series col6a = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", index, "Date");

            DataFrame df1 = new DataFrame(col1a, col2a, col3a, col4a, col5a, col6a);

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void SetColumn()
        {
            NDArray index = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"));

            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", index, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", index, "Date");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", index, "Date");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", index, "Date");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", index, "Date");
            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", index, "Date");

            DataFrame df1 = new DataFrame(col1, col2, col3, col4, col5, col6);

            Series t2 = new Series(new NDArray(15, 20, 30), "Low", index, "Date");

            df1["Low"] = t2;

            DataFrame t = df1["Low"];

            DataFrame a = new DataFrame(new Series(new NDArray(15, 20, 30), "Low", index, "Date"));

            Assert.True(t.Equals(a), "Arrays are not equal.");
        }


        [Fact]
        public void SetColumnNDArray()
        {
            NDArray index = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"));

            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", index, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", index, "Date");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", index, "Date");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", index, "Date");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", index, "Date");
            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", index, "Date");

            DataFrame df1 = new DataFrame(col1, col2, col3, col4, col5, col6);

            NDArray t2 = new NDArray(15, 20, 30);

            df1["Low"] = t2;

            DataFrame t = df1["Low"];

            DataFrame a = new DataFrame(new Series(new NDArray(15, 20, 30), "Low", index, "Date"));

            Assert.True(t.Equals(a), "Arrays are not equal.");
        }

        [Fact]
        public void TestPctChange()
        {
            Series col1 = new Series(new NDArray(100, 200, 300), "Column1");
            Series col2 = new Series(new NDArray(400, 500, 600), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            DataFrame t = df.PctChange();

            Series col5 = new Series(new NDArray(null, 1, 0.5), "Column1");
            Series col6 = new Series(new NDArray(null, 0.25, 0.2), "Column2");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestCumProd()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            DataFrame t = df.CumProd();

            Series col5 = new Series(new NDArray(10, 200, 6000), "Column1");
            Series col6 = new Series(new NDArray(40, 2000, 120000), "Column2");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestCumProdCol()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            DataFrame t = df["Column2"].CumProd();

            Series col6 = new Series(new NDArray(40, 2000, 120000), "Column2");

            DataFrame z = new DataFrame(col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestCumProdColNull()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(null, 50, 60), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            DataFrame t = df["Column2"].CumProd();

            Series col6 = new Series(new NDArray(null, 50, 3000), "Column2");

            DataFrame z = new DataFrame(col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestCumProdColNull2()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, null, 60), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            DataFrame t = df["Column2"].CumProd();

            Series col6 = new Series(new NDArray(40, null, 2400), "Column2");

            DataFrame z = new DataFrame(col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestSum()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            Series col3 = new Series(new NDArray(70, 80, 90), "Column1");
            Series col4 = new Series(new NDArray(100, 110, 120), "Column2");

            DataFrame df2 = new DataFrame(col3, col4);

            DataFrame t = df1 + df2;

            Series col5 = new Series(new NDArray(80, 100, 120), "Column1");
            Series col6 = new Series(new NDArray(140, 160, 180), "Column2");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestMinus()
        {
            Series col1 = new Series(new NDArray(100, 200, 300), "Column1");
            Series col2 = new Series(new NDArray(400, 500, 600), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            Series col3 = new Series(new NDArray(70, 80, 90), "Column1");
            Series col4 = new Series(new NDArray(100, 110, 120), "Column2");

            DataFrame df2 = new DataFrame(col3, col4);

            DataFrame t = df1 - df2;


            Series col5 = new Series(new NDArray(30, 120, 210), "Column1");
            Series col6 = new Series(new NDArray(300, 390, 480), "Column2");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestProd()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            Series col3 = new Series(new NDArray(10, 10, 10), "Column1");
            Series col4 = new Series(new NDArray(10, 10, 10), "Column2");

            DataFrame df2 = new DataFrame(col3, col4);

            DataFrame t = df1 * df2;

            Series col5 = new Series(new NDArray(100, 200, 300), "Column1");
            Series col6 = new Series(new NDArray(400, 500, 600), "Column2");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestDiv()
        {
            Series col1 = new Series(new NDArray(100, 200, 300), "Column1");
            Series col2 = new Series(new NDArray(400, 500, 600), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            Series col3 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col4 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df2 = new DataFrame(col3, col4);

            DataFrame t = df1 / df2;


            Series col5 = new Series(new NDArray(10, 10, 10), "Column1");
            Series col6 = new Series(new NDArray(10, 10, 10), "Column2");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestSumInt()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            DataFrame t = df1 + 10;

            Series col5 = new Series(new NDArray(20, 30, 40), "Column1");
            Series col6 = new Series(new NDArray(50, 60, 70), "Column2");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestMinusInt()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            DataFrame t = df1 - 10;

            Series col5 = new Series(new NDArray(0, 10, 20), "Column1");
            Series col6 = new Series(new NDArray(30, 40, 50), "Column2");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestProdInt()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            DataFrame t = df1 * 10;


            Series col5 = new Series(new NDArray(100, 200, 300), "Column1");
            Series col6 = new Series(new NDArray(400, 500, 600), "Column2");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestSetIndexJagged()
        {
            NDArray index = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"), DateTime.Parse("2018-03-15"), DateTime.Parse("2018-03-16"), DateTime.Parse("2018-03-17"));
            NDArray index1 = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"));
            NDArray index2 = new NDArray(DateTime.Parse("2018-03-15"), DateTime.Parse("2018-03-16"), DateTime.Parse("2018-03-17"));


            Series col1 = new Series(new NDArray(10, 20, 30), "Column1", index1);
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2", index2);
            Series col3 = new Series(index, "Date");

            DataFrame t = new DataFrame(col1, col2, col3);

            t.SetIndex("Date");

            Series col5 = new Series(new NDArray(10, 20, 30), "Column1", index1, "Date");
            Series col6 = new Series(new NDArray(40, 50, 60), "Column2", index2, "Date");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestGetValue()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            NDArray t = df1.GetValue();

            NDArray z = new NDArray(new NDArray(10, 20, 30), new NDArray(40, 50, 60));

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestSetCol()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            df1["Column3"] = new Series(new NDArray(70, 80, 90), "Column3");

            Series col5 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col6 = new Series(new NDArray(40, 50, 60), "Column2");
            Series col7 = new Series(new NDArray(70, 80, 90), "Column3");

            DataFrame z = new DataFrame(col5, col6, col7);

            Assert.True(df1.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestSetColNull()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            df1["Column3"] = null;

            Series col5 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col6 = new Series(new NDArray(40, 50, 60), "Column2");
            Series col7 = new Series(new NDArray(null, null, null), "Column3");

            DataFrame z = new DataFrame(col5, col6, col7);

            Assert.True(df1.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestSetColValue()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            df1["Column3"] = 10;

            Series col5 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col6 = new Series(new NDArray(40, 50, 60), "Column2");
            Series col7 = new Series(new NDArray(10, 10, 10), "Column3");

            DataFrame z = new DataFrame(col5, col6, col7);

            Assert.True(df1.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestSetColDataFrame()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            df1["Column3"] = new DataFrame(new Series(new NDArray(70, 80, 90), "Column3"));


            Series col5 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col6 = new Series(new NDArray(40, 50, 60), "Column2");
            Series col7 = new Series(new NDArray(70, 80, 90), "Column3");

            DataFrame z = new DataFrame(col5, col6, col7);

            Assert.True(df1.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestSetColNum()
        {
            Series col1 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col2 = new Series(new NDArray(40, 50, 60), "Column2");

            DataFrame df1 = new DataFrame(col1, col2);

            df1[1] = new Series(new NDArray(70, 80, 90), "Column2");

            Series col5 = new Series(new NDArray(10, 20, 30), "Column1");
            Series col6 = new Series(new NDArray(70, 80, 90), "Column2");

            DataFrame z = new DataFrame(col5, col6);

            Assert.True(df1.Equals(z), "Arrays are not equal.");
        }


        [Fact]
        public void TestIndexerGetLastRow()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            int i = df[-2, "Column2"];

            Assert.True(i.Equals(5), "Arrays are not equal.");
        }

        [Fact]
        public void TestIndexerGetNumLastRow()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            int i = df[-2, 1];

            Assert.True(i.Equals(5), "Arrays are not equal.");
        }

        [Fact]
        public void TestGetValueCol()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            int[] i = df.GetValueCol<int>("Column2");

            int[] t = new int[] { 4, 5, 6 };

            Assert.True(Enumerable.SequenceEqual(i, t), "Arrays are not equal.");
        }

        [Fact]
        public void TestSetNull()
        {
            NDArray index = new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14"));
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1", index, "Date");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2", index, "Date");

            DataFrame df = new DataFrame(col1, col2);

            df["Column3"] = null;
            df["Column4"] = null;

            Series cola = new Series(new NDArray(1, 2, 3), "Column1", index, "Date");
            Series colb = new Series(new NDArray(4, 5, 6), "Column2", index, "Date");
            Series colc = new Series(new NDArray(null, null, null), "Column3", index, "Date");
            Series cold = new Series(new NDArray(null, null, null), "Column4", index, "Date");

            DataFrame df1 = new DataFrame(cola, colb, colc, cold);


            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void TestGetRows()
        {
            Series col1 = new Series(new NDArray(1, 2, 3), "Column1");
            Series col2 = new Series(new NDArray(4, 5, 6), "Column2");

            DataFrame df = new DataFrame(col1, col2);

            NDArray index = new NDArray(2);

            Series cola = new Series(new NDArray(3), "Column1", index);
            Series colb = new Series(new NDArray(6), "Column2", index);

            DataFrame df1 = new DataFrame(cola, colb);

            DataFrame i = new DataFrame();

            foreach (DataFrame r in df)
            {
                i = r;
            }
               
            Assert.True(i.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void WriteCsv()
        {
            String filePath = GetPath("AMZN2", "csv");

            NDArray data = new NDArray(typeof(DateTime), typeof(double), typeof(double), typeof(double), typeof(double), typeof(double), typeof(int));

            Series col = new Series(new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14")), "Date", null, "Date");
            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", null, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", null, "Date");
            Series col3 = new Series(new NDArray(1586.699951, 1578.010010, 1590.890015), "Low", null, "Date");
            Series col4 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Close", null, "Date");
            Series col5 = new Series(new NDArray(1598.390015, 1588.180054, 1591.000000), "Adj Close", null, "Date");
            Series col6 = new Series(new NDArray(5174200, 6531900, 4175400), "Volume", null, "Date");

            DataFrame df = new DataFrame(col, col1, col2, col3, col4, col5, col6);

            df.SetIndex("Date");

            df.WriteCsv(filePath);

            FileStream fs = File.OpenRead(filePath);

            DataFrame df1 = DataFrame.LoadCsv(fs, ',', true, null, data, true);

            fs.Close();

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }

        [Fact]
        public void WriteCsvHeader()
        {
            String filePath = GetPath("AMZN3", "csv");

            NDArray data = new NDArray(typeof(DateTime), typeof(double), typeof(double));

            Series col = new Series(new NDArray(DateTime.Parse("2018-03-12"), DateTime.Parse("2018-03-13"), DateTime.Parse("2018-03-14")), "Date", null, "Date");
            Series col1 = new Series(new NDArray(1592.599976, 1615.959961, 1597.000000), "Open", null, "Date");
            Series col2 = new Series(new NDArray(1605.329956, 1617.540039, 1606.439941), "High", null, "Date");

            DataFrame df = new DataFrame(col, col1, col2);

            df.SetIndex("Date");

            df.WriteCsv(filePath, ',', true, new NDArray("Open", "High"));

            FileStream fs = File.OpenRead(filePath);

            DataFrame df1 = DataFrame.LoadCsv(fs, ',', true, null, data, true);

            fs.Close();

            Assert.True(df.Equals(df1), "Arrays are not equal.");
        }
    }      
}
