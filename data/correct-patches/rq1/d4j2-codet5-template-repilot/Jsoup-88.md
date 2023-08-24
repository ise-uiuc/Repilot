# Repilot Patch

```

        return val == null ? "" : val;
```

# Developer Patch

```
        return Attributes.checkNotNull(val);
```

# Context

```
--- bug/Jsoup-88/src/main/java/org/jsoup/nodes/Attribute.java

+++ fix/Jsoup-88/src/main/java/org/jsoup/nodes/Attribute.java

@@ -77,7 +77,8 @@

      @return the attribute value
      */
     public String getValue() {
-        return val;
+
+        return val == null ? "" : val;
     }
 
     /**
```

# Note

