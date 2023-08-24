# Repilot Patch

```

            if (rawCentralDirectoryData.length < expectedLength) {
```

# Developer Patch

```
            if (rawCentralDirectoryData.length < expectedLength) {
```

# Context

```
--- bug/Compress-19/src/main/java/org/apache/commons/compress/archivers/zip/Zip64ExtendedInformationExtraField.java

+++ fix/Compress-19/src/main/java/org/apache/commons/compress/archivers/zip/Zip64ExtendedInformationExtraField.java

@@ -256,7 +256,8 @@

                 + (hasCompressedSize ? DWORD : 0)
                 + (hasRelativeHeaderOffset ? DWORD : 0)
                 + (hasDiskStart ? WORD : 0);
-            if (rawCentralDirectoryData.length != expectedLength) {
+
+            if (rawCentralDirectoryData.length < expectedLength) {
                 throw new ZipException("central directory zip64 extended"
                                        + " information extra field's length"
                                        + " doesn't match central directory"
```

# Note

