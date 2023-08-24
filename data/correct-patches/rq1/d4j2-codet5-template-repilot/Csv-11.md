# Repilot Patch

```
final boolean emptyHeader = header == null || header.trim().isEmpty();
```

# Developer Patch

```
                    final boolean emptyHeader = header == null || header.trim().isEmpty();
```

# Context

```
--- bug/Csv-11/src/main/java/org/apache/commons/csv/CSVParser.java

+++ fix/Csv-11/src/main/java/org/apache/commons/csv/CSVParser.java

@@ -381,7 +381,7 @@

                 for (int i = 0; i < headerRecord.length; i++) {
                     final String header = headerRecord[i];
                     final boolean containsHeader = hdrMap.containsKey(header);
-                    final boolean emptyHeader = header.trim().isEmpty();
+final boolean emptyHeader = header == null || header.trim().isEmpty();
                     if (containsHeader && (!emptyHeader || (emptyHeader && !this.format.getIgnoreEmptyHeaders()))) {
                         throw new IllegalArgumentException("The header contains a duplicate name: \"" + header +
                                 "\" in " + Arrays.toString(headerRecord));
```

# Note

