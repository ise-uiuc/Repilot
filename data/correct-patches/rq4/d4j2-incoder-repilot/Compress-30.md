# Repilot Patch

```
        if (len == 0) {
            return 0;
        }
```

# Developer Patch

```
        if (len == 0) {
            return 0;
        }
```

# Context

```
--- bug/Compress-30/src/main/java/org/apache/commons/compress/compressors/bzip2/BZip2CompressorInputStream.java

+++ fix/Compress-30/src/main/java/org/apache/commons/compress/compressors/bzip2/BZip2CompressorInputStream.java

@@ -165,6 +165,9 @@

         if (this.in == null) {
             throw new IOException("stream closed");
         }
+        if (len == 0) {
+            return 0;
+        }
 
         final int hi = offs + len;
         int destOffs = offs;
```

# Note

