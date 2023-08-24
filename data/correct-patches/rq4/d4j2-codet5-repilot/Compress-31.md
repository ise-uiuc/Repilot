# Repilot Patch

```

```

# Developer Patch

```

```

# Context

```
--- bug/Compress-31/src/main/java/org/apache/commons/compress/archivers/tar/TarUtils.java

+++ fix/Compress-31/src/main/java/org/apache/commons/compress/archivers/tar/TarUtils.java

@@ -132,9 +132,8 @@

 
         for ( ;start < end; start++) {
             final byte currentByte = buffer[start];
-            if (currentByte == 0) {
-                break;
-            }
+
+
             // CheckStyle:MagicNumber OFF
             if (currentByte < '0' || currentByte > '7'){
                 throw new IllegalArgumentException(
```

# Note

