# Repilot Patch

```
//              LOG.debug("currentByte=" +currentByte);
```

# Developer Patch

```

```

# Context

```
--- bug/Compress-31/src/main/java/org/apache/commons/compress/archivers/tar/TarUtils.java

+++ fix/Compress-31/src/main/java/org/apache/commons/compress/archivers/tar/TarUtils.java

@@ -132,9 +132,7 @@

 
         for ( ;start < end; start++) {
             final byte currentByte = buffer[start];
-            if (currentByte == 0) {
-                break;
-            }
+//              LOG.debug("currentByte=" +currentByte);
             // CheckStyle:MagicNumber OFF
             if (currentByte < '0' || currentByte > '7'){
                 throw new IllegalArgumentException(
```

# Note

