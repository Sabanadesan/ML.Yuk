using System;
using System.Collections.Generic;
using System.Text;

namespace ML.Yuk
{
    public static class Utility
    {
        public static void PrettyPrint(DataFrame df)
        {
            var sb = new StringBuilder();
            int width = GetLongestValueLength(df) + 4;

            for (int i = 0; i < df.Columns.Length; i++)
            {
                NDArray cols = df.Columns;
                var name = cols[i];

                // Left align by 10
                sb.Append(string.Format(name.PadRight(width)));
            }

            sb.AppendLine();

            long numberOfRows = Math.Min(df.Length, 25);
            for (int i = 0; i < numberOfRows; i++)
            {
                for (int j = 0; j < df.Columns.Length; j++)
                {
                    var value = df[i, j];
                    sb.Append((value ?? "null").ToString().PadRight(width));
                }

                sb.AppendLine();
            }

            Console.WriteLine(sb.ToString());
        }

        private static int GetLongestValueLength(DataFrame df)
        {
            long numberOfRows = Math.Min(df.Length, 25);
            int longestValueLength = 0;

            for (int i = 0; i < numberOfRows; i++)
            {
                for (int j = 0; j < df.Columns.Length; j++)
                {
                    var value = df[i, j];
                    longestValueLength = Math.Max(longestValueLength, value?.ToString().Length ?? 0);
                }

            }

            return longestValueLength;
        }
    }

    public partial class TextFieldParser
    {
        public TextFieldParser()
        {
        }

        public string[] ParseFields(string text)
        {
            string[] parts = text.Split(',');

            List<string> newParts = new List<string>();
            bool inQuotes = false;
            string currentPart = string.Empty;

            for (int i = 0; i < parts.Length; i++)
            {
                string part = parts[i];
                inQuotes = (inQuotes || part.StartsWith("\""));
                if (inQuotes)
                {
                    currentPart = (string.IsNullOrEmpty(currentPart)) ? part : string.Format("{0},{1}", currentPart, part);
                }
                else
                {
                    currentPart = part;
                }
                inQuotes = (inQuotes && !part.EndsWith("\""));
                if (!inQuotes)
                {
                    currentPart = currentPart.Replace("\"", "");
                    newParts.Add(currentPart);
                    currentPart = string.Empty;
                }
            }

            return newParts.ToArray();
        }
    }
}
